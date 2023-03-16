f=open("chatbot_data.txt","r+")
data=f.read()
if "hello" not in data:
     f.write("hello")
