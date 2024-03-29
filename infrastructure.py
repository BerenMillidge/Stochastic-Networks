
import smtplib
from email import *
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import datetime


credential_file = 'credentials.txt'
credential_spliter = '$$$'

f = open(credential_file, "r")
credential_string = f.read()
f.close()

#get variables
from_email, password, to_email = credential_string.split(credential_spliter)

PORT = 465 # arbitrarily from stack overflow


# I'll also put the logging capabilities here! for now!

log_file = './logs.txt'

# write these as they become necessary
def format_results_log(results):
	# no obvious formatting can be applied
	if isinstance(results, str):
		return results
	elif isinstance(results, dict):
		s = ""
		for k,v in results.items():
			s += str(k) + ":" + str(v) + "\n"
		return s
	elif hasattr(results, "__iter__") or hasattr(results, "__getitem__"):
		# so it is an iterable of some sort
		s = "Results list: \n"
		for res in results:
			s += str(res) + "\n"
		s += "\n"
		return s
	else:
		print("Did not know how to format results of type " + str(type(results)))
		return s 

def datestring():
	now = datetime.datetime.now()
	return str(now)

def format_traceback_exception_log(e, info, traceback):
	msg = "EXCEPTION: " + datestring() + " "
	msg = msg + str(e) + "\n"
	type_, value_, tb_ = info
	msg += str(type_) + " " + str(value_) + "\n"
	msg += str(traceback) + "\n"
	return msg

def format_exception_log(e):
	msg = "EXCEPTION: " + datestring() + " "
	msg = msg + str(e) + "\n"
	return msg

def format_message_string(message):
	msg = "MESSAGE: " + datestring()+ " "
	msg += str(message)
	msg += "\n"
	return message


def log(message):
	f = open(log_file, 'a+')
	f.write(message)
	f.close()



def send_mail(subject, message, port=PORT):
	# just recreate the server every time... because! it's probaly really bad... but doesn't matter!
	try:
		msg = MIMEMultipart()
		msg['Subject'] = subject
		msg['From'] = from_email
		msg['To'] = to_email
		msg.attach(MIMEText(message, 'plain'))
		server = smtplib.SMTP_SSL('smtp.gmail.com', port) # presumably arbitrary port number
		server.ehlo()
		server.login(from_email, password)
		server.sendmail(from_email, [to_email], msg.as_string())
		server.close()
		print("Message Sent!")
		log(format_message_string("Email Sent"))
	except Exception as e:
		print("Connection failed \n")
		print(e)
		log(format_exception_log(e))


