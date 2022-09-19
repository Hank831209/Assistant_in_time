import datetime
from email.mime import image
from sqlite3 import DateFromTicks
import time
from flask import Flask
# from __future__ import unicode_literals
from flask import request, abort, render_template, session, redirect, url_for
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import *
from inference import model
from flask_wtf import FlaskForm
from wtforms import StringField, DateTimeField, SelectField, StringField, TextAreaField, SubmitField, IntegerField
from wtforms.validators import DataRequired
import requests
import json
import configparser
import os
import shutil
# from wtforms.fields.html5 import DateField
# from urllib import parse


app = Flask(__name__)

# 把機密資料寫成txt來讀取, 主要只會用到channelAccessToken和channelSecret
secretFile = json.load(open("secretFile.txt",'r'))
channelAccessToken = secretFile['channelAccessToken']
channelSecret = secretFile['channelSecret']

line_bot_api = LineBotApi(channelAccessToken)
handler = WebhookHandler(channelSecret)

# 標準寫法基本不會改
@app.route("/", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']
    # get request body as text
    body = request.get_data(as_text=True)
    # 紀錄收到過的訊息, 一般不會放
    app.logger.info("Request body: " + body)
    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        print("Invalid signature. Please check your channel access token/channel secret.")
        abort(400)
    return 'OK'


# def saveImg(event):
#     # r'./yolov5_6_2/data/images'  # 圖片存到這路徑, 必須.jpg結尾
#     if event.message.type == image:
#         image_content = line_bot_api.get_message_content(event.message.id)
#         image_name = image_name.upper() + '.jpg'  # image_name??
#         path = './statics/' + image_name  
#         with open(path, 'wb') as fd:
#             for chunk in image_content.iter_content():
#                 fd.write(chunk)

#     model(image_name)
#     text = '系統辨識中...'
#     message = TextSendMessage(text=text)
#     line_bot_api.reply_message(event.reply_token, message)

#     # 解析完刪除資料夾
#     result_path= 'yolov5_6_2/runs/detect/exp'
#     # time.sleep(1)
#     if os.path.isdir(result_path):
#         shutil.rmtree(result_path)



@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    text = event.message.text + '我好帥'
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=text))


@handler.add(MessageEvent)
def handle_message(event):
    if (event.message.type == "image"):
        start_time = time.time()
        style_type = {
            0: 'Baby', 
            1: 'Princess',
            2: 'Casual Wear',
            3: 'Gentleman'
        }

        SendImage = line_bot_api.get_message_content(event.message.id)
        img_name = str(event.message.id)  # 圖片名稱
        path_img_save = 'yolov5_6_2/data/images/'
        path_result_dir = 'yolov5_6_2/runs/detect/exp'
        path_img_file = os.path.join(path_img_save, img_name + '.jpg')

        if not os.path.exists(path_img_save):
            os.mkdir(path_img_save)
            
        # 存圖片
        with open(path_img_file, 'wb') as file:
            for img in SendImage.iter_content():
                file.write(img)

        # 模型偵測
        style_list = model(img_name, path='yolov5_6_2/runs/detect/exp', 
                        path_weights='best_test.pt', num_classes=4, net='large')
        print('style_list:\t', style_list)
        if not style_list:
            text = '未偵測到有人'
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=text))
        else:
            reply_style_list = list()
            for style in style_list:
                reply_style_list.append(TextSendMessage(text=style_type[style]))
            line_bot_api.reply_message(event.reply_token, reply_style_list)

        # 刪除運行完物件偵測的圖片
        if os.path.isdir(path_result_dir):
            shutil.rmtree(path_result_dir)

        # 刪除原始圖片
        if os.path.isfile(path_img_file):
            os.remove(path_img_file)

        end_time = time.time()
        print('Total Time:\t', end_time - start_time)




if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)