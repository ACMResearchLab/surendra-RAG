
from transformers import RobertaTokenizer, T5ForConditionalGeneration

old_text = """
def svg_to_image(string, size=None):
    if isinstance(string, unicode):
        string = string.encode('utf-8')
        renderer = QtSvg.QSvgRenderer(QtCore.QByteArray(string))
    if not renderer.isValid():
        raise ValueError('Invalid SVG data.')
    if size is None:
        size = renderer.defaultSize()
        image = QtGui.QImage(size, QtGui.QImage.Format_ARGB32)
        painter = QtGui.QPainter(image)
        renderer.render(painter)
    return image

"""

if __name__ == '__main__':
    tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base-multi-sum')
    model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base-multi-sum')

    text = """
def dliauwhdialuwhdi():
    word_list = ["apple", "banana", "orange", "grape", "kiwi"]
    return random.choice(word_list)


    """

    input_ids = tokenizer(text, return_tensors="pt").input_ids

    generated_ids = model.generate(input_ids, max_length=100)
    print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
    # this prints: "Convert a SVG string to a QImage."