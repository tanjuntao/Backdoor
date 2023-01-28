import json

from flask import Flask, request

app = Flask("LinkeFL Flask Service")


@app.route("/", methods=["GET", "POST"])
def hello_world():
    return "Hello world from Flask service."


@app.route("/log", methods=["POST"])
def logging():
    msg = request.form["msg"]  # logger message is stored with the 'msg' key
    msg = msg.strip()
    print(msg)

    if msg[0] == "{" and msg[-1] == "}":  # means that msg is a stringfied json
        pass
        # metrics = json.loads(msg)  # metrics is a Python dict
        # TODO: deal with metrics here

    else:  # normal logger message
        pass
        # normal_msg = msg
        # TODO: deal with normal message here

    return msg


if __name__ == "__main__":
    app.run(host="localhost", port=5000, debug=True)
