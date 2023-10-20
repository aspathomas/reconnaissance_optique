import os
from flask import Flask, request, make_response, jsonify, send_from_directory
from service.processingPicture import ProcessingPicture
from functools import wraps

app = Flask(__name__)

#-------------------------------------------#
# process
@app.route('/processing', methods=['POST'])
def process():
    print('test')
    file = request.files['file']
    processPicture = ProcessingPicture()
    return processPicture.process(file)

if __name__ == '__main__':
    #define the localhost ip and the port that is going to be used
    # in some future article, we are going to use an env variable instead a hardcoded port 
    app.run(host='0.0.0.0', port=os.getenv('PORT'))