import pyqb



class Pyqb():

	def __init__(self, domain_name, username, password):


       # def Client(url="http://www.quickbase.com", database=None, proxy=None, user_token=None):
       self.qbc = pyqb.Client(url=domain_name)
       # Below authenticate is not required if `user_token` argument is passed to pyqb.Client() above
       self.qbc.authenticate(username='myusername', password='mypassword')


    def doQuery(query=None, qid=None, qname=None, database=None, fields=None, fmt=False, rids=False, sort_fields=None, options=False):
       self.qbc.doquery(qid)
       self.qbc.doquery(query, database)
       self.qbc.doquery(qid, fields, fmt, rids)


    def editRecord(rid=None, database=None, fields=None, update_id=None):
       f = { "6": "newvalue" }
       res = self.qbc.editrecord(rid, database, fields)


    def addRecord(database=None, fields=None):
       f = { "hostname": "myhost", "7": "1.2.3.4" }
       res = self.qbc.addrecord(database, fields)


    def deleteRecord(rid=None, database=None):
       res = self.qbc.deleterecord(rid, database)

