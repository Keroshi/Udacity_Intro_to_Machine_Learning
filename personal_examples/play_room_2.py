# import numpy as np
#
# arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# arr.tolist()

# # Sample text
# text = "This is a sample text that will be split into words."
#
# # Split the text into words
# words = text.split()
#
# # Print the resulting list of words
# print(words)
# print(text)

# # Original text
# text = "Once upon a time in a quiet village nestled between rolling hills, there was a small community where everyone knew each other by name. The village was known for its beautiful gardens, where vibrant flowers bloomed in a riot of colors and the air was always filled with their sweet fragrance. The villagers were a close-knit group, always ready to lend a helping hand to one another. They shared their joys and sorrows, celebrated festivals together, and supported each other through difficult times. Among them was an old man named Henry, who was known for his wisdom and kindness. He had a little house at the edge of the village with a garden that was the envy of everyone. Henry loved to spend his days tending to his plants and sharing stories with the children who visited him. One summer, a severe drought hit the village. The rivers and wells dried up, and the gardens began to wither. The villagers were worried about their crops and the lack of water. Henry, however, remained calm and hopeful. He gathered the villagers and encouraged them to dig a new well at a spot he believed would have water. Despite their doubts, they trusted Henry and worked together to dig the well. After days of hard work, they finally struck water. The village rejoiced, and the gardens began to flourish again. The villagers realized the importance of unity and perseverance. From that day on, they looked up to Henry not just for his wisdom but also for his unwavering faith and the spirit of community he had nurtured. The village continued to thrive, and the story of the new well became a tale passed down through generations, reminding everyone of the strength that comes from working together and believing in each other."
# text_array = text.split()
# formatted_text = ''
#
# # List of punctuations to remove
# removal_words = ["a", "the", "was", "and"]
#
# # Remove punctuation and reassemble the text
# for word in text_array:
#     if word in removal_words:
#         word = word.replace(word, '')
#     formatted_text += word + " "
#
# # Print the resulting text
# print(formatted_text.strip())

import numpy as np
import IPython.display as display
from matplotlib import pyplot as plt
import io
import base64

ys = 200 + np.random.randn(100)
x = [x for x in range(len(ys))]

fig = plt.figure(figsize=(4, 3), facecolor='w')
plt.plot(x, ys, '-')
plt.fill_between(x, ys, 195, where=(ys > 195), facecolor='g', alpha=0.6)
plt.title("Sample Visualization", fontsize=10)

data = io.BytesIO()
plt.savefig(data)
image = F"data:image/png;base64,{base64.b64encode(data.getvalue()).decode()}"
alt = "Sample Visualization"
display.display(display.Markdown(F"""![{alt}]({image})"""))
plt.close(fig)
