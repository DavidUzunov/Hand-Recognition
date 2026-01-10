from flask import Flask, render_template, jsonify

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/ping')
def ping():
    return jsonify({'message': 'Pong!', 'status': 'success'})

if __name__ == '__main__':
    app.run(debug=True)
