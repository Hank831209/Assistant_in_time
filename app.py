import time
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import *
from inference import model, delete_dir
import json
import os
from flask import Flask, abort, render_template, request, session, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import SubmitField, SelectField, HiddenField, StringField
from wtforms.validators import DataRequired
from wtforms.fields.html5 import DateField
from datetime import datetime
import re


app = Flask(__name__)

# 把機密資料寫成txt來讀取, 主要只會用到channelAccessToken和channelSecret
secretFile = json.load(open("secretFile.txt",'r'))
channelAccessToken = secretFile['channelAccessToken']
channelSecret = secretFile['channelSecret']

line_bot_api = LineBotApi(channelAccessToken)
handler = WebhookHandler(channelSecret)


# 標準寫法基本不會改
@app.route("/callback", methods=['POST'])
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


# @handler.add(MessageEvent, message=TextMessage)
# def handle_message(event):
#     text = event.message.text + '我好帥'
#     line_bot_api.reply_message(event.reply_token, TextSendMessage(text=text))


@handler.add(MessageEvent, message=ImageMessage)
def handle_message(event):
    start_time = time.time()
    style_type = {
        0: '服飾風格：幼兒服', 
        1: '服飾風格：公主風',
        2: '服飾風格：休閒風',
        3: '服飾風格：紳士風'
    }

    SendImage = line_bot_api.get_message_content(event.message.id)
    img_name = str(event.message.id)  # 圖片名稱
    path_img_dir = './static'
    result_dir = './detect/exp'
    path_img = os.path.join(path_img_dir, img_name + '.jpg')

    if not os.path.exists(path_img_dir):
        os.mkdir(path_img_dir)
    if not os.path.exists('./detect'):
        os.mkdir('./detect')

    # 存圖片
    with open(path_img, 'wb') as file:
        for img in SendImage.iter_content():
            file.write(img)

    # 模型偵測
    style_list = model(path_img, yolo=True)
                
    if len(style_list) <= 1:  # 一人以下
        predict_result = model(path_img, yolo=False)
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=style_type[predict_result]))
        time.sleep(1)
        delete_dir(path_img, result_dir)
        print('Already delete')
    else:  # 多人照片
        reply_list = list()
        for style in style_list:
            reply_list.append(TextSendMessage(text=style_type[style]))
        line_bot_api.reply_message(event.reply_token, reply_list)
        time.sleep(1)
        delete_dir(path_img, result_dir)
        print('Already delete')
    end_time = time.time()
    print('Total Time:\t', end_time - start_time)


# 資料庫
# Flask-WTF requires an enryption key - the string can be anything
app.config['SECRET_KEY']='mykey'

# Flask-Bootstrap requires this line
Bootstrap(app)

# app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:008488@34.83.236.232:3306/reserve'
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:008488@35.192.97.51:3306/project'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# this variable, db, will be used for all SQLAlchemy commands
db = SQLAlchemy(app)


_option_id = {
    (r'純數位檔方案 $4,999'): 1,
    (r'紀念方案 ＄8,888'): 2,
    (r'典藏方案 $13,500'): 3,
    (r'全檔精緻方案 $18,500'): 4,
    (r'成長方案-2年內拍攝3次（可包含新生兒寫真）$19,999'): 5
}

_plus = {
    (r'無'): 1,
    (r'外加入鏡 $500/人(爸媽以外成員)'): 2,
    (r'媽媽另加妝髮造型 $1,000/套'): 3,
    (r'外景拍攝另加 $500-3,000（依地點報價）'): 4,
    (r'新生兒造型 $1,000/套'): 5,
    (r'外景拍攝另加 $500-3,000（依地點報價）'): 6
}

class Member(db.Model):
    __tablename__ = 'member'
    member_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(25), nullable=False)
    tel = db.Column(db.String(14), nullable=False)
    user_id = db.Column(db.String(50), nullable=False)

    # db_Member_Reserve = db.relationship("Reserve", backref="member")
    
    def __init__(self, name, tel, user_id):
        self.name = name
        self.tel = tel
        self.user_id = user_id
        

class Scheme(db.Model):
    __tablename__ = 'scheme'
    option_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    _class = db.Column(db.String(10), nullable=False)
    name = db.Column(db.String(20), nullable=False)
    price = db.Column(db.Integer, nullable=False)
    content = db.Column(db.String(100), nullable=False)

    # db_Scheme_Reserve = db.relationship("Reserve", backref="scheme")
    
    def __init__(self, option_id, _class, name, price, content):
        self.option_id = option_id
        self._class = _class
        self.name = name
        self.price = price
        self.content = content
        
        
class Reserve(db.Model):
    __tablename__ = 'reserve'
    reserve_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(25), nullable=False)
    tel = db.Column(db.String(14), nullable=False)
    num_people = db.Column(db.String(5), nullable=False)
    reserve_date = db.Column(db.String(30), nullable=False)  # Check
    plus = db.Column(db.String(100), nullable=False)
    record_date = db.Column(db.String(30), nullable=False)  # Check
    
    user_id = db.Column(db.String(50), nullable=False)
    option = db.Column(db.Integer, nullable=False)
    
    # option = db.Column(db.Integer, db.ForeignKey('scheme.option_id'))
    # user_id = db.Column(db.String(50), db.ForeignKey('member.user_id'))

    def __init__(self, user_id, name, tel, num_people, reserve_date, option, plus, record_date):
        self.user_id = user_id
        self.name = name
        self.tel = tel
        self.num_people = num_people
        self.reserve_date = reserve_date
        self.option = option
        self.plus = plus
        self.record_date = record_date

class Reserve_form(FlaskForm):
    name = StringField('預約姓名', validators=[DataRequired()])
    tel = StringField('預約電話', validators=[DataRequired()])
    num_people = SelectField('預約人數', choices=[('1'), ('2'), ('3'), ('4'), ('5'), ('5人以上')])
    reserve_date =  DateField('預約日期', format='%Y-%m-%d')
    AM_PM = SelectField('預約時間', choices=[('上午'), ('下午')])
    option = SelectField('兒童寫真方案選擇', choices=[choice for choice in _option_id.keys()])
    plus = SelectField('加購項目', choices=[choice for choice in _plus.keys()])
    submit = SubmitField("確認")


@app.route('/', methods=['GET','POST'])
def index():
    """首頁"""
    form = Reserve_form()
    form_data = dict()  # 感謝表單的回傳值
    Error = list()  # 錯誤訊息
    if form.validate_on_submit():  # 傳入重複的會報錯
        form_data['name'] = request.form.get('name')                             # sss <class 'str'>
        form_data['tel'] = request.form.get('tel')                               # 0987635241 <class 'str'>
        form_data['num_people'] = request.form.get('num_people')                         # 1 <class 'str'>
        form_data['reserve_date'] = request.form.get('reserve_date')             # 2022-10-15 <class 'str'>
        form_data['AM_PM'] = request.form.get('AM_PM')                           # 下午 <class 'str'>
        form_data['option'] = request.form.get('option')                         # 紀念方案 ＄8,888 <class 'str'>
        form_data['plus'] = request.form.get('plus')                             # 外加入鏡 $500/人(爸媽以外成員) <class 'str'>
        # ----------------------------------------------------------------------------------------------------------------------------------
        
        # 電話處理
        check1 = re.search(pattern=r'^09\d{2}-?\d{3}-?\d{3}$', string=form_data['tel'])  # 手機
        check2 = re.search(pattern=r'^\d{2}-?\d{4}-?\d{4}$', string=form_data['tel'])  # 市話
        check3 = re.search(pattern=r'^\d{3}-?\d{3}-?\d{4}$', string=form_data['tel'])  # 市話
        if (not check1) and (not check2) and (not check3):
            Error.append('預約電話輸入失敗 請重新填寫')

        # 預約日期處理 防止重複
        reserve_date = (form_data['reserve_date'] + ' AM' if form_data['AM_PM'] == '上午' else form_data['reserve_date'] + ' PM')
        if Reserve.query.filter_by(reserve_date=reserve_date).first():  # 是否該時段已被預約
            Error.append('日期輸入失敗 請重新填寫')
        
        # 選擇方案處理
        option = _option_id[form_data['option']]
        user_id = '123'
        if Error:  # 資料有誤
            return render_template('reserve.html', form=form, Error=Error)
        else:      # 資進行完處理後存入資料庫(預約資料)
            record = Reserve(user_id=user_id, name=form_data['name'], tel=form_data['tel'], 
                            num_people=form_data['num_people'], reserve_date=reserve_date, 
                            option=option, plus=form_data['plus'], 
                            record_date=datetime.now().strftime('%Y-%m-%d %H:%M:%S %p'))
            db.session.add(record)
            db.session.commit()
            if not Member.query.filter_by(user_id=user_id).first(): 
                member = Member(name=form_data['name'], tel=form_data['tel'], user_id=user_id)
                db.session.add(member)
                db.session.commit()
            return render_template('thankyou.html', form_data=form_data)
    else:
        # Error.append('輸入失敗 請重新填寫')  # 錯誤訊息
        return render_template('reserve.html', form=form, Error=Error)  # 保持留在該頁面


# @handler.add(MessageEvent, message=TextMessage)
# def handle_message(event):
#     if event.message.text == '預約表單':
#         user_id = event.source.user_id
#         url_user_id = 'https://2c9a-114-43-185-141.jp.ngrok.io' + f'/{user_id}'
#         line_bot_api.reply_message(reply_token=event.reply_token, messages=TextSendMessage(text=url_user_id))


@app.route('/create')
def create():
    # Create data
    db.create_all()
    return 'ok'


if __name__ == "__main__":
    # app.run(host='0.0.0.0', debug=True)
    app.run()