# test script to practice sending email...
"""
import smtplib
from email.MIMEMultipart import MIMEMultipart
from email.MIMEText import MIMEText

fromaddr = "email"
toaddr = "email"
msg = MIMEMultipart()
msg["From"] = fromaddr
msg["To"] = toaddr
msg["Subject"] = "Python email test"
body = "Python test mail script"
msg.attach(MIMEText(body, 'plain'))

server = smtplib.SMTP('')
"""


import smtplib

# Import the email modules we'll need
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Open a plain text file for reading.  For this example, assume that
# the text file contains only ASCII characters.
#fp = open(textfile, 'rb')
# Create a text/plain message
#msg = MIMEText(fp.read())
#fp.close()

# me == the sender's email address
# you == the recipient's email address
me = 'from email'
you = 'to email'
gmail_password = "pass"

msg = MIMEMultipart()
msg['Subject'] = 'Email Test'
msg['From'] = me
msg['To'] = you
body = "From Python test email script"
msg.attach(MIMEText(body, 'plain'))

# Send the message via our own SMTP server, but don't include the
# envelope header.
#s = smtplib.SMTP('localhost')
#s.sendmail(me, [you], msg.as_string())
#s.quit()

# setup the google server
try:
	server = smtplib.SMTP_SSL('smtp.gmail.com', 465) # presumably arbitrary port number
	server.ehlo()
	server.login(me, gmail_password)
	server.sendmail(me, [you], msg.as_string())
	server.close()
	print("Message Sent!")
except Exception as e:
	print("Connection failed \n")
	print(e)


# that's god it sholdn't hut performance wrapping all my code inside a try-catch block... which is really fantastic and nice
# so that's goodto know as well... I'll have it sendme the code in the exception if there is one, and regular results updates
# so that is nice... now to turn this into an actual function!
# I cuold also have it write to a logfile if anything happened!
# let's try and write a script with a logfile and everything to do this in the proper way, which might be a fun way to learn!