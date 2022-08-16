from trycourier import Courier

def send_mail(name,email,phone,option,subject,message,owner_mail):

    if option == "1":
            t = "Bug"
    elif option == "2":
            t = "Inappropriate content"
    elif option == "3":
            t = "Suggestion"            
    elif option == "4":
            t = "Other"
        
    body = 'Name : ' + name + '\nEmail : ' + email + '\nPhone : ' \
           + phone + '\nType : ' + t + "\n\nMessage : " + message
           
    client = Courier(auth_token="pk_prod_MP2J8SYNT1MKXFM4Q6XTMW27DRZZ")
    resp = client.send_message(
    message={
        "to": {
        "email": owner_mail
        },
        "content": {
        "title": subject,
        "body": body
        }
    }
    )
    print('Mail Sent')
