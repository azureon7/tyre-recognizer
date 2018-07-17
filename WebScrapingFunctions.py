import requests
from bs4 import BeautifulSoup


def Parser(Marca, Param1, Param2, Param3):
    Param1 = str(Param1)
    Param2 = str(Param2)
    Param3 = str(Param3)
    parametri1 = {'cart_id': 'hffmk3Cw5Fkp5I2F.130.858511787',\
                'Cookie': 'sea_rd_it_Google_Search',\
                'Breite': Param1,\
                'Felge': Param3,\
                'Herst': Marca,\
                'kategorie': '6',\
                'Quer': Param2,\
                'ranzahl': '4',\
                'rsmFahrzeugart': 'ALL',\
                'suchen': 'Trova+pneumatici',\
                'sort_by': 'preis',\
                'desname': 'gommadiretto.it',\
                'shop': 'RDIT',\
                'currency': 'EUR',\
                'Ang_pro_Seite': '50',\
                'weiter':'0'}
    parametri2 = {'cart_id': 'hffmk3Cw5Fkp5I2F.130.858511787',\
                'Cookie': 'sea_rd_it_Google_Search',\
                'Breite': Param1,\
                'Felge': Param3,\
                'Herst': Marca,\
                'kategorie': '6',\
                'Quer': Param2,\
                'ranzahl': '4',\
                'rsmFahrzeugart': 'ALL',\
                'suchen': 'Trova+pneumatici',\
                'sort_by': 'preis',\
                'desname': 'gommadiretto.it',\
                'shop': 'RDIT',\
                'currency': 'EUR',\
                'Ang_pro_Seite': '50',\
                'weiter':'0'}

    page = requests.get('https://www.gommadiretto.it/cgi-bin/rshop.pl?dsco=130', params = parametri1)
    soup = BeautifulSoup(page.text, "html.parser")
    form = soup.select('form.pure-form')[0]
    righe = form.find_all('div', attrs={'class': 'row'})
    '''
    page = requests.get('https://www.gommadiretto.it/cgi-bin/rshop.pl?dsco=130', params = parametri2)
    soup = BeautifulSoup(page.text, "html.parser")
    form = soup.select('form.pure-form')[0]
    righe+= form.find_all('div', attrs={'class': 'row'})
    '''
    listone = []
    for i in range(50):
        low = i*2
        upp = i*2+2
        righe1 = form.find_all('div', attrs={'class': 'row'})[low:upp]
        nome1 = righe1[0].select('div.col-xs-12')[0].text
        nome1 = nome1.replace('\n', ' ').strip()
        url1 = 'https://www.gommadiretto.it'+righe1[0].find_all('a')[1]['href']
        if Marca not in nome1:
            break
        img1 = righe1[1].select('img.img-serp')[0]['src']
        try:
            price1 = righe1[1].select('div.serp_price')[0].text.strip().split()[1]
        except:
            price1 = righe1[1].select('div.price')[0].text.strip().split()[1]
        listone.append({'nome': nome1, 'img': img1, 'price': price1, 'url': url1})
    
    output = (listone[0], listone[-1])
    return output