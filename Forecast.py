import datetime
today = datetime.date.today()
a = today + datetime.timedelta(days=-today.weekday(), weeks=0)
print(a)