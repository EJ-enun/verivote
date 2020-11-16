
import os



## App settings
name = "VeriVote NG" 

host = "0.0.0.0"

port = int(os.environ.get("PORT", 5000))

debug = False

contacts = "https://www.linkedin.com/in/enun-enun-13b99519a/"

code = "https://github.com/EJ-enun/verivote"

fontawesome = 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css'



## File system
root = os.path.dirname(os.path.dirname(__file__)) + "/"



## DB
