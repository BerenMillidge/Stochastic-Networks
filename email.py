
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


credential_file = './credentials.txt'
credential_spliter = '$$$'

f = open(credential_file, "r")
credential_string = f.read()
f.close()

#get variables
from_email, password, to_email = credential_string.split(credential_spliter)


# I'll also put the logging capabilities here! for now!

log_file = './logs.txt'

def log(message):
	f = open(log_file, 'a+')


def send_mail(subject, message):
	# just recreate the server every time... because! it's probaly really bad... but doesn't matter!
	try:
		msg = MIMEMultipart()
		msg['Subject'] = subject
		msg['From'] = me
		msg['To'] = you
		msg.attach(MIMEText(message, 'plain'))
		server = smtplib.SMTP_SSL('smtp.gmail.com', 465) # presumably arbitrary port number
		server.ehlo()
		server.login(from_email, password)
		server.sendmail(me, [to_email], msg.as_string())
		server.close()
		print("Message Sent!")
	except Exception as e:
		print("Connection failed \n")
		print(e)
		# log something here!


