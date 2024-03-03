
from cos_similarity import *

# TODO: write some functions that do similar things


similar_functions = [
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


# TODO: write some functions that do different things

different_functions = [
    # go
    """
func sumOfSquaresOfEven(nums []int) int {
	sum := 0
	for _, num := range nums {
		if num%2 == 0 {
			sum += num * num
		}
	}
	return sum
}
    """,

    # python
    """
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)
    """,

    # ruby
    """
def palindrome?(str)
  str.downcase == str.downcase.reverse
end
    """,

    # javascript
    """
function averageOfPositive(nums) {
    let count = 0;
    let sum = 0;
    for (const num of nums) {
        if (num > 0) {
            sum += num;
            count++;
        }
    }
    return count === 0 ? 0 : sum / count;
}
    """

]

similar_functions_embeddings = [
    pl_embedding_without_nl(func) for func in similar_functions]
different_functions_embeddings = [
    pl_embedding_without_nl(func) for func in different_functions]


print(f"the values following should be similar, in semanticness")
for x in range(0, len(similar_functions)):
    for y in range(x, len(similar_functions)):
        cos_similarity = kevin_cos_sim(
            similar_functions_embeddings[x], similar_functions_embeddings[y])
        print(cos_similarity)


print(f"the values following should be non-similar, in semanticness")
for x in range(0, len(different_functions)):
    for y in range(x, len(different_functions)):
        cos_similarity = kevin_cos_sim(
            different_functions_embeddings[x], different_functions_embeddings[y])
        print(cos_similarity)
