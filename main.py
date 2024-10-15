from flask import Flask, render_template, request, jsonify
from handle_text import predict_rating

app = Flask(__name__)


def handle_text(text):
    rating = predict_rating(text)
    return f"Automatically generated rating: {rating}"


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/execute', methods=['POST'])
def execute():
    text = request.json.get('text')
    try:
        result = handle_text(text)
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'result': f'Error: {str(e)}'})


if __name__ == '__main__':
    app.run(debug=True)
