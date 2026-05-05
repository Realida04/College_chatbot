from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

from store_index import rag_chain

app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        message = data.get("message")

        completion = rag_chain.invoke({"input": message})

        print("DEBUG:", completion)

        # Safe extraction
        reply = completion.get("answer") or completion.get("result") or str(completion)

        return jsonify({"response": reply})

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)