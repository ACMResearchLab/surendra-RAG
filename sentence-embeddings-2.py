from transformers import AutoTokenizer, AutoModel
import torch


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    # First element of model_output contains all token embeddings
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(
        -1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Sentences we want sentence embeddings for
sentences = [
    # in go
    """
    func findMax(nums []int) int {
    if len(nums) == 0 {
        return -1 // Assuming -1 denotes an error or no maximum found
    }
    max := nums[0]
    for _, num := range nums {
        if num > max {
            max = num
        }
    }
    return max
}
    """,

    # in python
    """
def find_max(nums):
    if not nums:
        return None
    return max(nums)
    """,

    # in ruby
    """
def find_max(nums)
  return nil if nums.empty?
  nums.max
end
    """,

    # in javascript
    """
function findMax(nums) {
    if (nums.length === 0) {
        return -1; // Assuming -1 denotes an error or no maximum found
    }
    return Math.max(...nums);
}
    """,
]

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('mchochlov/codebert-base-cd-ft')
model = AutoModel.from_pretrained('mchochlov/codebert-base-cd-ft')

# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True,
                          truncation=True, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)

# Perform pooling. In this case, max pooling.
sentence_embeddings = mean_pooling(
    model_output, encoded_input['attention_mask'])

print("Sentence embeddings:")
print(sentence_embeddings)
print(sentence_embeddings.shape)


tensors = torch.split(sentence_embeddings, split_size_or_sections=1, dim=0)
print(len(tensors))
cos = torch.nn.CosineSimilarity(dim=1)

print(f"the values following should be similar, in semanticness")
for x in range(0, len(tensors)):
    for y in range(x, len(tensors)):
        cos_similarity = cos(tensors[x], tensors[y])
        print(cos_similarity)