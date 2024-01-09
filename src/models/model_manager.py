import torch
import torch.nn as nn

#from transformers.modeling_outputs import BaseModelOutput

#from models.multimodal.molt5 import MolT5
from models import SUPPORTED_MOL_GRAPH_ENCODER, SUPPORTED_MOL_SMILES_ENCODER, SUPPORTED_TEXT_ENCODER, SUPPORTED_IMAGE_ENCODER, SUPPORTED_DECODER, SUPPORTED_Tokenizer, SUPPORTED_CKPT
from models.multimodal import Modal_Fusion
from utils.xutils import print_model_info
from transformers.modeling_outputs import BaseModelOutput
import torch.nn.functional as F

def pad_tensors_to_max_length(tensors, max_len, device, dim=0):
    """Pad each tensor in the list to the specified max_len."""
    padded_tensors = []
    for tensor in tensors:
        pad_size = max_len - tensor.size(dim)
        padded_tensor = torch.cat([tensor, torch.zeros(*tensor.shape[:-1], pad_size).to(device)], dim=dim)
        padded_tensors.append(padded_tensor)
    return padded_tensors

class MLP(nn.Module):
    #task_num 1 2 12 27 617
    def __init__(self, input_size, hidden_size, output_size, num_layers=4, dropout_prob=0.2, decay_factor=0.5):
        super(MLP, self).__init__()

        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        # Add (num_layers - 2) hidden layers (because the first layer and the output layer are always there)
        current_size = hidden_size
        for _ in range(num_layers - 2):
            next_size = int(current_size * decay_factor)
            layers.append(nn.Linear(current_size, next_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_prob))
            current_size = next_size
    
        layers.append(nn.Linear(current_size, output_size))  # Output layer
    
            # Create the sequential model
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x

class MolModel(nn.Module):
    def __init__(self, config):
        super(MolModel, self).__init__()
        text_modals = ['SMILES', 'IUPAC', 'InChI', 'SELFIES', 'caption']
        self.config = config
        self.config.hidden_size = 768
        #print(config)
        self.input_modal = config.input_modal.split(',')
        self.output_modal = config.output_modal
        self.modals = config.input_modal.split(',')
        self.modals.append(self.output_modal)
        #print(f"self.input_modal : {self.input_modal}")
        if(self.output_modal == 'caption'):
            self.min_length = 30
        else:
            self.min_length = 4
        self.prompt = config.prompt
        if("graph" in self.input_modal):
            self.graph2d_encoder = SUPPORTED_MOL_GRAPH_ENCODER[config.graph_encoder](config.graph_encoder).graph_encoder
            #graph_ckpt = SUPPORTED_CKPT[config.graph_encoder]
            #graph_ckpt = torch.load(graph_ckpt, map_location='cpu')
            #self.graph2d_encoder.load_state_dict(graph_ckpt['encoder'], strict=False)
            """
            for param in self.graph2d_encoder.parameters():
                param.requires_grad = False
            """
            self.num_features = 300
            self.graph_fc_hidden = nn.Linear(self.num_features, self.config.hidden_size)

        if(self.config.text_encoder is not None and self.config.text_encoder != ''):
            text_encoder_config = {'ckpt':SUPPORTED_CKPT[config.text_encoder]}
            self.text_encoder = SUPPORTED_TEXT_ENCODER[self.config.text_encoder](text_encoder_config)
                
        if("image" in self.input_modal):
            if(config.image_encoder=='swin'):
                self.image_encoder = SUPPORTED_IMAGE_ENCODER[config.image_encoder]()
                image_ckpt = SUPPORTED_CKPT[config.image_encoder]
                image_ckpt = torch.load(image_ckpt, map_location='cpu')
                self.image_encoder.load_state_dict(image_ckpt['encoder'], strict=False)
                self.num_features = 1536
                self.image_fc_hidden = nn.Linear(self.num_features, self.config.hidden_size)
            else:
                image_encoder_config = {'ckpt':SUPPORTED_CKPT[config.image_encoder]}
                self.image_encoder = SUPPORTED_IMAGE_ENCODER[config.image_encoder](image_encoder_config)
                
            if(config.image_encoder=='swin_nopre'):
                self.num_features = 768
                self.image_fc_hidden = nn.Linear(self.num_features, self.config.hidden_size)
            elif(config.image_encoder=='resnet'):
                self.num_features = 2048
                self.image_fc_hidden = nn.Linear(self.num_features, self.config.hidden_size)
            elif(config.image_encoder=='vit'):
                self.num_features = 768
                self.image_fc_hidden = nn.Linear(self.num_features, self.config.hidden_size)
                
        self.input_modal_num = len(self.input_modal)
        if(self.input_modal_num)>1:
            self.fusion_model = Modal_Fusion(self.config)
        
        self.all_in_text_encoder = self.all_in_text_encoder()
        if(self.all_in_text_encoder == True):
            #print("xxx")
            self.decoder = None
        else:
            #print("xxxx")
            self.decoder = SUPPORTED_DECODER[config.decoder]()
            
        if(self.prompt == None or self.prompt == ""):
            pass
        else:
            self.prompt_tokenizer = SUPPORTED_Tokenizer[config.decoder].from_pretrained(SUPPORTED_CKPT[config.decoder])
            self.prompt = f" is the {self.config.input_modal} of the molecule, and {self.config.output_modal} is :"
            self.prompt = self.prompt_tokenizer(text=self.prompt, return_tensors="pt")

        if(self.config.task_name == 'mpp'):
            if(self.config.task_num == 0):
                task_num = 1
            else:
                task_num = self.config.task_num
            self.property_mlp = MLP(768,512,task_num, self.config.mlp_layers_num, self.config.dropout)
            
    def get_mpp_embeddings(self, h):
        h = h.transpose(1, 2) 
        if(self.config.pool=='avg'):
            h = F.avg_pool1d(h, h.shape[2]).squeeze(2) 
        else:
            h = F.max_pool1d(h, h.shape[2]).squeeze(2) 
        return h
                
        
        
    def forward(self, mol):
        h, h_attention_mask = self.encode_h(mol)
        labels = {}
        #labels["input_ids"] = mol[f"{self.output_modal}_labels"]["input_ids"].masked_fill(~mol[f"{self.output_modal}_labels"]["attention_mask"].bool(), -100)
        labels["input_ids"] = mol[f"{self.output_modal}_labels"]["input_ids"]
        labels["attention_mask"] = mol[f"{self.output_modal}_labels"]["attention_mask"]
        if(self.all_in_text_encoder == False):
            h = BaseModelOutput(
                last_hidden_state= h,
                hidden_states= None,
                attentions= None
            )
            output = self.decoder(
                encoder_outputs=h,
                attention_mask=h_attention_mask,
                decoder_attention_mask=mol[f"{self.output_modal}_labels"]["attention_mask"],
                labels=labels["input_ids"]
            )
        elif(self.all_in_text_encoder == True):
            if(self.config.decoder == 'gptneo'):
                mol_text = self.mol_text(mol)
                output = self.text_encoder(
                    input_ids = mol_text['input_ids'],
                    attention_mask=mol_text['attention_mask'],
                    decoder_attention_mask=mol[f"{self.output_modal}_labels"]["attention_mask"],
                    labels=labels["input_ids"]
                )
            else:
                h = BaseModelOutput(
                    last_hidden_state= h,
                    hidden_states= None,
                    attentions= None
                )
                output = self.text_encoder(
                    encoder_outputs = h,
                    attention_mask=h_attention_mask,
                    decoder_attention_mask=mol[f"{self.output_modal}_labels"]["attention_mask"],
                    labels=labels["input_ids"]
                )
        else:
            output = None
        return output
    
    def get_attentions(self,mol):
        h, h_attention_mask = self.encode_h(mol)
        labels = {}
        #labels["input_ids"] = mol[f"{self.output_modal}_labels"]["input_ids"].masked_fill(~mol[f"{self.output_modal}_labels"]["attention_mask"].bool(), -100)
        labels["input_ids"] = mol[f"{self.output_modal}_labels"]["input_ids"]
        labels["attention_mask"] = mol[f"{self.output_modal}_labels"]["attention_mask"]
        if(self.all_in_text_encoder == False):
            h = BaseModelOutput(
                last_hidden_state= h,
                hidden_states= None,
                attentions= None
            )
            output = self.decoder.get_attentions(
                encoder_outputs=h,
                attention_mask=h_attention_mask,
                decoder_attention_mask=mol[f"{self.output_modal}_labels"]["attention_mask"],
                labels=labels["input_ids"],
                output_attentions = True
            )
        elif(self.all_in_text_encoder == True):
            if(self.config.decoder == 'gptneo'):
                mol_text = self.mol_text(mol)
                output = self.text_encoder.get_attentions(
                    input_ids = mol_text['input_ids'],
                    attention_mask=mol_text['attention_mask'],
                    decoder_attention_mask=mol[f"{self.output_modal}_labels"]["attention_mask"],
                    labels=labels["input_ids"],
                    output_attentions = True
                )
            else:
                h = BaseModelOutput(
                    last_hidden_state= h,
                    hidden_states= None,
                    attentions= None
                )
                output = self.text_encoder.get_attentions(
                    encoder_outputs = h,
                    attention_mask=h_attention_mask,
                    decoder_attention_mask=mol[f"{self.output_modal}_labels"]["attention_mask"],
                    labels=labels["input_ids"],
                    output_attentions = True
                )
        else:
            output = None
        return output
    
    def forward_mpp(self, mol):
        h, h_attention_mask = self.encode_h(mol)
        h = self.get_mpp_embeddings(h)
        result = self.property_mlp(h)
 
        return result

    def forward_mtr(self, mol):
        encode_mtr_vector = self.encode_mtr_vector(mol)
        v1 = encode_mtr_vector['input_modal']
        v2 = encode_mtr_vector['output_modal']

        # v1-v2 similarity: [batch_size, batch_size]
        sim_v1_v2 = torch.matmul(v1, v2.t())

        # v2-v1 similarity: [batch_size, batch_size]
        sim_v2_v1 = torch.matmul(v2, v1.t())

        # Assuming you have batch-wise labels for matching
        targets = torch.arange(v1.size(0)).to(v1.device)

        # Calculate the cross-entropy losses
        loss_v1_v2 = F.cross_entropy(sim_v1_v2, targets)
        loss_v2_v1 = F.cross_entropy(sim_v2_v1, targets)

        # Final loss
        loss = (loss_v1_v2 + loss_v2_v1) / 2.0

        return loss
        
    
    def all_in_text_encoder(self):
        no_decoder = False
        if(self.config.text_encoder == self.config.decoder):
            no_decoder = True
        elif(self.config.decoder == '' or self.config.decoder == None):
            no_decoder = True
        
        return no_decoder
            

    def encode_h(self, mol):
        embeddings = self.encode_embeddings(mol)
        #labels = mol[f"{self.output_modal}_labels"]
        if(self.input_modal_num)>1:
            if('graph' in self.input_modal):
                graph_embeddings = embeddings['graph']
            if('image' in self.input_modal):
                image_embeddings = embeddings['image']
            embeddings.pop('graph', None)
            embeddings.pop('image', None) 
            if(len(embeddings) > 0):
                text_embeddings = sum(embeddings.values())
                if('graph' in self.input_modal):
                    h, h_attention_mask = self.fusion_model(text_embeddings, graph_embeddings)
                elif('image' in self.input_modal):
                    h, h_attention_mask = self.fusion_model(text_embeddings, image_embeddings)
                else:
                    h = text_embeddings
                    h_attention_mask = torch.ones(h.size()[:-1], dtype=torch.long, device=h.device)
            else:
                print("Graph structure and image do not need to enter at the same time")
                h = graph_embeddings
                h_attention_mask = torch.ones(h.size()[:-1], dtype=torch.long, device=h.device)

        else:
            h = embeddings[self.input_modal[0]]
            h_attention_mask = torch.ones(h.size()[:-1], dtype=torch.long, device=h.device)

        if(self.prompt == None or self.prompt == ""):
            pass
        elif(self.config.task_name in ['molopt2smi','molopt2IUPAC']):
            prompt_all = []
            batch_size = h.size(0)
            for i in range(batch_size):
                combined = {}
                #prompt = f"is the scaffold of this molecule, property: {self.config.target_property}, original: {mol['org_property'][i]}, target:{mol['target_property'][i]}"
                prompt = f"is the scaffold of this molecule, original: {mol['org_property'][i]}, target:{mol['target_property'][i]}, please generate the target molecule SMILES:"
                prompt_embeds = self.prompt_tokenizer(prompt, return_tensors="pt").to(h.device)
                for key in prompt_embeds:
                    #print(f"mol['scaffold'][key][i]: {mol['scaffold'][key][i]}")
                    #print(f"prompt_embeds[key][0]: {prompt_embeds[key][0]}")
                    combined[key] = torch.cat([mol['scaffold'][key][i], prompt_embeds[key][0]], dim=0)
                prompt_all.append(combined)
            all_embeds = [item['input_ids'] for item in prompt_all]
            all_attention_masks  = [item['attention_mask'] for item in prompt_all]
            
            # Determine max tensor length for all_embeds and all_attention_masks
            max_embed_length = max([tensor.size(0) for tensor in all_embeds])
            max_mask_length = max([tensor.size(0) for tensor in all_attention_masks])
            max_tensor_length = max(max_embed_length, max_mask_length)

            # Pad each tensor in the lists to the maximum tensor length
            all_embeds = pad_tensors_to_max_length(all_embeds, max_tensor_length, h.device)
            all_attention_masks = pad_tensors_to_max_length(all_attention_masks, max_tensor_length, h.device)

            # Now, you can stack them
            prompt_embeds = torch.stack(all_embeds, dim=0)
            attention_mask = torch.stack(all_attention_masks, dim=0).to(h.device)
            prompt_embeds = prompt_embeds.long().to(h.device)
            
            if(self.all_in_text_encoder == True):
                if(self.config.decoder in ['t5','t511','molt5', 'molt5_large']):
                    prompt_embeds = self.text_encoder.main_model.get_input_embeddings()(prompt_embeds)
                elif(self.config.text_encoder == 'gpt2'):
                    prompt_embeds = self.text_encoder.main_model.transformer.wte(prompt_embeds)
                    #h = self.text_encoder.fc(h)
                elif(self.config.text_encoder == 'biogpt'):
                    prompt_embeds = self.text_encoder.main_model.biogpt.embed_tokens(prompt_embeds)
                    h = self.text_encoder.fc(h)
            else:
                if(self.config.decoder in ['t5','t511','molt5','molt5_large']):
                    prompt_embeds = self.decoder.main_model.get_input_embeddings()(prompt_embeds)
                elif(self.config.decoder == 'gpt2'):
                    prompt_embeds = self.decoder.main_model.transformer.wte(prompt_embeds)
                    #h = self.text_encoder.fc(h)
                elif(self.config.decoder == 'biogpt'):
                    prompt_embeds = self.text_encoder.main_model.biogpt.embed_tokens(prompt_embeds)
                    h = self.text_encoder.fc(h)
                
            #print(f"h: {h.shape}")
            #print(f"prompt_embeds: {prompt_embeds.shape}")
            h = torch.cat([h, prompt_embeds.to(h.device)], dim=1)
            #print(f"h_attention_mask: {h_attention_mask.shape}")
            #print(f"attention_mask: {attention_mask.shape}")
            h_attention_mask = torch.cat([h_attention_mask, attention_mask.to(h.device)], dim=1)
        else:
            input_ids = self.prompt['input_ids']
            attention_mask = self.prompt['attention_mask']
            batch_size = h.size(0)
            prompt = {}
            prompt_embeds = input_ids.repeat(batch_size, 1).to(h.device)
            attention_mask = attention_mask.repeat(batch_size, 1).to(h.device)
            if(self.all_in_text_encoder == True):
                if(self.config.decoder in ['t5','t511','molt5']):
                    prompt_embeds = self.text_encoder.main_model.get_input_embeddings()(prompt_embeds)
                elif(self.config.text_encoder == 'gpt2'):
                    prompt_embeds = self.text_encoder.main_model.transformer.wte(prompt_embeds)
                    #h = self.text_encoder.fc(h)
            else:
                if(self.config.decoder in ['t5','t511','molt5']):
                    prompt_embeds = self.decoder.main_model.get_input_embeddings()(prompt_embeds)
                elif(self.config.decoder == 'gpt2'):
                    prompt_embeds = self.decoder.main_model.transformer.wte(prompt_embeds)
                    #h = self.text_encoder.fc(h)
            #attention_mask = torch.ones_like(prompt['input_ids'])
            h_return = torch.cat([h, prompt_embeds.to(h.device)], dim=1)
            if(h_return.shape[1]<512):
                h_attention_mask = torch.cat([h_attention_mask, attention_mask.to(h.device)], dim=1)
            else:
                h_return = h
            h = h_return
        return h, h_attention_mask

    def encode_mpp_h(self, mol):
        embeddings = self.encode_embeddings(mol)
        #labels = mol[f"{self.output_modal}_labels"]
        if(self.input_modal_num)>1:
            if('graph' in self.input_modal):
                graph_embeddings = embeddings['graph']
            embeddings.pop('graph', None)
            if(len(embeddings) > 0):
                text_embeddings = sum(embeddings.values())
                if('graph' in self.input_modal):
                    h, h_attention_mask = self.fusion_model(text_embeddings, graph_embeddings)
                else:
                    h = text_embeddings
                    h_attention_mask = torch.ones(h.size()[:-1], dtype=torch.long, device=h.device)
            else:
                print("Graph structure and image do not need to enter at the same time")
                h = graph_embeddings
                h_attention_mask = torch.ones(h.size()[:-1], dtype=torch.long, device=h.device)

        else:
            h = embeddings[self.input_modal[0]]
            h_attention_mask = torch.ones(h.size()[:-1], dtype=torch.long, device=h.device)
        return h, h_attention_mask
            
    
    def encode_embeddings(self, mol):
        embeddings = {}
        if("graph" in self.input_modal):
            graph_embeddings = self.graph_encode(mol)
            embeddings['graph'] = graph_embeddings
        if("SMILES" in self.input_modal):
            smiles_embeddings = self.text_encode(mol)
            embeddings['SMILES'] = smiles_embeddings
        if("caption" in self.input_modal):
            caption_embeddings = self.text_encode(mol)
            embeddings['caption'] = caption_embeddings
        if("IUPAC" in self.input_modal):
            iupac_embeddings = self.text_encode(mol)
            embeddings['IUPAC'] = iupac_embeddings
        if("SELFIES" in self.input_modal):
            selfies_embeddings = self.text_encode(mol)
            embeddings['SELFIES'] = selfies_embeddings
        if("InChI" in self.input_modal):
            inchi_embeddings = self.text_encode(mol)
            embeddings['InChI'] = inchi_embeddings
        if("image" in self.input_modal):
            image2d_embeddings = self.image_encode(mol)
            embeddings['image'] = image2d_embeddings
            
        return embeddings
    
    def pool_h(self,pool,h):
        h = h.transpose(1, 2) 
        if(pool=='avg'):
            h = F.avg_pool1d(h, h.shape[2]).squeeze(2) 
        else:
            h = F.max_pool1d(h, h.shape[2]).squeeze(2) 
        return h
            
    def encode_mtr_vector(self, mol):
        vectors = {}
        embeddings = self.encode_mtr_embeddings(mol)
        h_in = embeddings["input_modal"]
        h_out = embeddings["output_modal"]
        
        h_in = self.pool_h(self.config.pool,h_in)
        h_out = self.pool_h(self.config.pool_out,h_out)

        vectors["input_modal"] = h_in
        vectors["output_modal"] = h_out
        #print(f"vectors:{vectors}")
        return vectors
    
    def encode_mtr_embeddings(self, mol):
        embeddings = {}
        if(self.config.input_modal in ['SMILES', 'IUPAC', 'InChI', 'SELFIES', 'caption']):
            text_embeddings = self.text_mtr_encode(mol,'input')  
            embeddings["input_modal"] = text_embeddings
        if(self.config.output_modal in ['SMILES', 'IUPAC', 'InChI', 'SELFIES', 'caption']):
            text_embeddings = self.text_mtr_encode(mol,'output')  
            embeddings["output_modal"] = text_embeddings
        if("graph" == self.config.input_modal):
            graph_embeddings = self.graph_encode(mol)
            embeddings["input_modal"] = graph_embeddings
        if("image" == self.config.input_modal):
            image2d_embeddings = self.image_encode(mol)
            embeddings["input_modal"] = image2d_embeddings
            
        return embeddings
    
    def image_encode(self, mol):
        image2ds = mol['image']
        image2d_embeddings = self.image_encoder(image2ds)
        #print(f"image2d_embeddings : {image2d_embeddings}")
        #print(f"image2d_embeddings.shape : {image2d_embeddings.shape}")
        if(self.config.image_encoder=='resnet'):
            batch_size = image2d_embeddings.size(0)
            image2d_embeddings = image2d_embeddings.view(batch_size, 7 * 7, 2048)
        image2d_embeddings = self.image_fc_hidden(image2d_embeddings)
        return image2d_embeddings
    
    def graph_encode(self, mol):
        graph_feats, node_feats, node_feats_mask = self.graph2d_encoder(mol["graph"])
        node_feats = self.graph_fc_hidden(node_feats)
        return node_feats
    
    def text_mtr_encode(self, mol, input_output):
        if(input_output == 'input'):
            h = self.text_encoder.encode(mol[self.config.input_modal])
        else:
            modal = f"{self.config.output_modal}_labels"
            h = self.text_encoder.encode(mol[modal])
        return h
    
    def text_encode(self, mol):
        if('SMILES' in self.input_modal):
            h = self.text_encoder.encode(mol['SMILES'])
            #h_attention_mask = torch.ones(h.size()[:-1], dtype=torch.long, device=h.device)
        if('caption' in self.input_modal):
            h = self.text_encoder.encode(mol['caption'])
            #h_attention_mask = torch.ones(h.size()[:-1], dtype=torch.long, device=h.device)
        if('IUPAC' in self.input_modal):
            h = self.text_encoder.encode(mol['IUPAC'])
        if('SELFIES' in self.input_modal):
            h = self.text_encoder.encode(mol['SELFIES'])
        if('InChI' in self.input_modal):
            h = self.text_encoder.encode(mol['InChI'])
            
        return h
    
    def mol_text(self, mol):
        if('SMILES' in self.input_modal):
            mol_text = mol['SMILES']
            #h_attention_mask = torch.ones(h.size()[:-1], dtype=torch.long, device=h.device)
        if('caption' in self.input_modal):
            mol_text = mol['caption']
            #h_attention_mask = torch.ones(h.size()[:-1], dtype=torch.long, device=h.device)
        if('IUPAC' in self.input_modal):
            mol_text = mol['IUPAC']
        if('SELFIES' in self.input_modal):
            mol_text = mol['SELFIES']
        if('InChI' in self.input_modal):
            mol_text = mol['InChI']
            
        return mol_text
        
    def generate_text(self, mol):
        #input_ids = mol[self.input_modal]['input_ids']
        #attention_mask = mol[self.input_modal]['attention_mask']
        

        if(self.all_in_text_encoder == True):
            if(self.config.decoder == 'gptneo'):
                mol_text = self.mol_text(mol)
                text = self.text_encoder.decode(
                    input_ids = mol_text['input_ids'],
                    attention_mask = mol_text['attention_mask'], 
                    num_beams = 5,
                    max_length = 512
                )
            else:
                h, h_attention_mask = self.encode_h(mol)
                h = BaseModelOutput(
                    last_hidden_state= h,
                    hidden_states= None,
                    attentions= None
                )
                text = self.text_encoder.decode(
                    encoder_outputs = h, 
                    encoder_attention_mask = h_attention_mask, 
                    num_beams = 5,
                    max_length = 512
                )
        else:
            h, h_attention_mask = self.encode_h(mol)
            h = BaseModelOutput(
                last_hidden_state= h,
                hidden_states= None,
                attentions= None
            )
            text = self.decoder.decode(
                encoder_outputs = h, 
                encoder_attention_mask = h_attention_mask, 
                num_beams = 5,
                max_length = 512
            )
        return text