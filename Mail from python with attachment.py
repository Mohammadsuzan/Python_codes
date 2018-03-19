# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 11:16:06 2018

@author: Mohammadsuzan.Shaikh
"""

def send_mail_with_attachment(send_from,send_to,subject,text,server,port,username='',password='',isTls=True):
    msg = MIMEMultipart()
    msg['From'] = send_from
    msg['To'] = send_to
    msg['Date'] = formatdate(localtime = True)
    msg['Subject'] = subject
    msg.attach(MIMEText(text))

    part = MIMEBase('application', "octet-stream")
    part.set_payload(open("D:\\Projects\\12. Case tracker\\Data\\Perfios generated and not scored.xlsx", "rb").read())
    encoders.encode_base64(part)
    
    '''File to be attached'''
    filename='Perfios generated and not scored.xlsx'

    part.add_header('Content-Disposition', 'attachment; filename=%s' % filename)
    msg.attach(part)

    #context = ssl.SSLContext(ssl.PROTOCOL_SSLv3)
    #SSL connection only working on Python 3+
    smtp = smtplib.SMTP(server, port)
    if isTls:
        smtp.starttls()
    smtp.login(username,password)
    smtp.sendmail(send_from, send_to, msg.as_string())
    smtp.quit()