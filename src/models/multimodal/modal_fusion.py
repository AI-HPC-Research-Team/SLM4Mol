import torch
import torch.nn as nn
import torch.nn.functional as F


class Modal_Fusion(nn.Module):
    def __init__(self, config = None):
        super(Modal_Fusion, self).__init__()
        if(config == None or config.fusion_net == 'add'):
            self.model = Add_Fusion()
        elif(config.fusion_net == 'weight_add'):
            self.model = Weight_Fusion(config.hidden_size)
        elif(config.fusion_net == 'self_attention'):
            self.model = SelfAttentionFusion(config.hidden_size)
        else:
            self.model = Add_Fusion()
            
    def forward(self, features_text, features_graph):
        fused_embedding = self.model(features_text, features_graph)
        return fused_embedding
                 

class Add_Fusion(nn.Module):
    def __init__(self):
        super(Add_Fusion, self).__init__()

    def forward(self, features_text, features_graph):
        # Determine the maximum sequence length in the batch
        max_seq_len = max(features_text.size(1), features_graph.size(1))

        # Create attention masks
        attention_mask_text = torch.ones(features_text.size(0), features_text.size(1)).to(features_text.device)
        attention_mask_graph = torch.ones(features_graph.size(0), features_graph.size(1)).to(features_graph.device)

        # Pad the features and update attention masks
        if features_text.size(1) < max_seq_len:
            padding_size = max_seq_len - features_text.size(1)
            attention_mask_text = torch.cat([attention_mask_text, torch.zeros(features_text.size(0), padding_size).to(features_text.device)], dim=1)
            features_text = torch.cat([features_text, torch.zeros(features_text.size(0), padding_size, features_text.size(2)).to(features_text.device)], dim=1)
        
        if features_graph.size(1) < max_seq_len:
            padding_size = max_seq_len - features_graph.size(1)
            attention_mask_graph = torch.cat([attention_mask_graph, torch.zeros(features_graph.size(0), padding_size).to(features_graph.device)], dim=1)
            features_graph = torch.cat([features_graph, torch.zeros(features_graph.size(0), padding_size, features_graph.size(2)).to(features_graph.device)], dim=1)

        # Merge the two attention masks
        combined_attention_mask = attention_mask_text * attention_mask_graph

        # Compute fused embeddings by adding the features directly
        fused_embedding = features_text + features_graph

        return fused_embedding, combined_attention_mask        
        
class Weight_Fusion(nn.Module):
    def __init__(self, input_size):
        super(Weight_Fusion, self).__init__()
        self.fc = nn.Linear(2 * input_size, 1)  # double the size because of concatenation

    def forward(self, features_text, features_graph):
        # Determine the maximum sequence length in the batch
        max_seq_len = max(features_text.size(1), features_graph.size(1))

        # Create attention masks
        attention_mask_text = torch.ones(features_text.size(0), features_text.size(1)).to(features_text.device)
        attention_mask_graph = torch.ones(features_graph.size(0), features_graph.size(1)).to(features_graph.device)

        # Pad the features and update attention masks
        if features_text.size(1) < max_seq_len:
            padding_size = max_seq_len - features_text.size(1)
            attention_mask_text = torch.cat([attention_mask_text, torch.zeros(features_text.size(0), padding_size).to(features_text.device)], dim=1)
            features_text = torch.cat([features_text, torch.zeros(features_text.size(0), padding_size, features_text.size(2)).to(features_text.device)], dim=1)
        
        if features_graph.size(1) < max_seq_len:
            padding_size = max_seq_len - features_graph.size(1)
            attention_mask_graph = torch.cat([attention_mask_graph, torch.zeros(features_graph.size(0), padding_size).to(features_graph.device)], dim=1)
            features_graph = torch.cat([features_graph, torch.zeros(features_graph.size(0), padding_size, features_graph.size(2)).to(features_graph.device)], dim=1)

        # Merge the two attention masks
        combined_attention_mask = attention_mask_text * attention_mask_graph

        # Compute adaptive weights
        combined = torch.cat([features_text, features_graph], dim=-1)
        weights = torch.sigmoid(self.fc(combined))

        # Compute fused embeddings
        fused_embedding = weights * features_text + (1 - weights) * features_graph

        return fused_embedding, combined_attention_mask

class SelfAttentionFusion(nn.Module):
    def __init__(self, input_size, attention_heads=1):
        super(SelfAttentionFusion, self).__init__()

        self.input_size = input_size
        self.num_heads = attention_heads
     
        self.query = nn.Linear(self.input_size, self.input_size * self.num_heads)
        self.key = nn.Linear(self.input_size, self.input_size * self.num_heads)
        self.value = nn.Linear(self.input_size, self.input_size * self.num_heads)
        
        self.fc_out = nn.Linear(self.input_size * self.num_heads, self.input_size)

    def forward(self, features_text, features_graph):
        # Create attention masks
        attention_mask_text = torch.ones(features_text.size(0), features_text.size(1)).to(features_text.device)
        attention_mask_graph = torch.ones(features_graph.size(0), features_graph.size(1)).to(features_graph.device)

        max_seq_len = max(features_text.size(1), features_graph.size(1))
        if features_text.size(1) < max_seq_len:
            padding_size = max_seq_len - features_text.size(1)
            attention_mask_text = torch.cat([attention_mask_text, torch.zeros(features_text.size(0), padding_size).to(features_text.device)], dim=1)
            features_text = torch.cat([features_text, torch.zeros(features_text.size(0), padding_size, features_text.size(2)).to(features_text.device)], dim=1)
        
        if features_graph.size(1) < max_seq_len:
            padding_size = max_seq_len - features_graph.size(1)
            attention_mask_graph = torch.cat([attention_mask_graph, torch.zeros(features_graph.size(0), padding_size).to(features_graph.device)], dim=1)
            features_graph = torch.cat([features_graph, torch.zeros(features_graph.size(0), padding_size, features_graph.size(2)).to(features_graph.device)], dim=1)

        
        combined_features = features_text + features_graph
        combined_attention_mask = attention_mask_text * attention_mask_graph
        
        
        Q = self.query(combined_features)
        K = self.key(combined_features)
        V = self.value(combined_features)

        
        Q = Q.view(combined_features.size(0), combined_features.size(1), self.num_heads, self.input_size).permute(0, 2, 1, 3)
        K = K.view(combined_features.size(0), combined_features.size(1), self.num_heads, self.input_size).permute(0, 2, 1, 3)
        V = V.view(combined_features.size(0), combined_features.size(1), self.num_heads, self.input_size).permute(0, 2, 1, 3)

        
        attention_score = torch.matmul(Q, K.permute(0, 1, 3, 2)) / (self.input_size ** 0.5)
        attention_probs = torch.nn.functional.softmax(attention_score, dim=-1)

        
        weighted_values = torch.matmul(attention_probs, V)
        weighted_values = weighted_values.permute(0, 2, 1, 3).contiguous()

        
        fused_embedding = self.fc_out(weighted_values.view(combined_features.size(0), combined_features.size(1), -1))

        return fused_embedding, combined_attention_mask