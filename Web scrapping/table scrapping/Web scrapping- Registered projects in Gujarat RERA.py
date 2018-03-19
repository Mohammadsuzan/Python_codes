import urllib2
from bs4 import BeautifulSoup

pge="https://gujrera.gujarat.gov.in/registeredProjectList?offset=0&maxResults=10000000"
page=urllib2.urlopen(pge)

soup=BeautifulSoup(page)

#soup.prettify()

all_tables=soup.find_all('table')
right_table=soup.find('table',class_="reportTable")

A=[]
B=[]
C=[]
D=[]
for row in right_table.findAll("tr"):
    cells=row.find_all("td")
    if len(cells)==4:
        A.append(cells[0].find(text=True))
        B.append(cells[1].find(text=True))
        C.append(cells[2].find(text=True))
        D.append(cells[3].find(text=True))

report_table=pd.DataFrame(A,columns=['Registeration Number'])        
report_table['Project Name']=B
report_table['Type of Project']=C
report_table['District']=D

report_table.to_csv('E:\\My Python codes\\Web scrapping- Registered projects in Gujarat RERA.csv')
