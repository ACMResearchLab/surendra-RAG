from sentence_transformers import SentenceTransformer
code_fragments = [
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

model = SentenceTransformer('mchochlov/codebert-base-cd-ft')
embeddings = model.encode(code_fragments)
print(embeddings)
print(embeddings.shape)
print(type(embeddings))
