import requests
from bs4 import BeautifulSoup
import base64
import re
import json
import os
import shutil
from selenium import webdriver 
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException
from urllib.parse  import unquote
from urllib.parse import quote

location_driver = './chromedriver.exe'

class ImageSearchEngine(object):
  def __init__(self, k=20, engine='baidu', data_path='data/image/'):
    self.proxy_addrs = {
      'http': 'http://127.0.0.1:7890',
      'https': 'https://127.0.0.1:7890',
    }
    self.search_patterns = {
      'google': 'https://www.google.com/search?q={}&source=lnms&tbm=isch',
      'baidu': 'https://image.baidu.com/search/index?tn=baiduimage&ie=utf-8&word={}&rn={}',
      'bing': 'https://cn.bing.com/images/search?q={}',
      'yahoo': 'https://images.search.yahoo.com/search/images;?p={}',
      'aol': 'https://search.aol.com/aol/image;?q={}',
      'duckduckgo': 'https://duckduckgo.com/?q={}&iax=images&ia=images'
    }
    
    self.k = k
    self.engine = engine
    self.data_path = data_path
    self.driver = None
    self.searchEngine = {
      'baidu': self.get_topk_from_baidu,
      'google': self.get_topk_from_google,
      'bing': self.get_topk_from_bing,
      'yahoo': self.get_topk_from_yahoo,
      'aol': self.get_topk_from_yahoo,
      'duckduckgo': self.get_topk_from_duckduckgo
    }

  def start_brower(self):
    chrome_options = Options()
    chrome_options.add_argument("--disable-infobars")
    chrome_options.add_argument('--proxy-server=127.0.0.1:7890')
    driver = webdriver.Chrome(executable_path=location_driver, chrome_options=chrome_options)
    driver.maximize_window()  
    jsCode = "var q=document.documentElement.scrollTop=100000"
    driver.execute_script(jsCode)
    driver.execute_script(jsCode)
    driver.execute_script(jsCode)
    driver.execute_script(jsCode)
    return driver

  def get_topk_from_baidu(self, engine, html, s, _class, folder_name):
    image_lis = html.find_all('li', 'imgitem')
    for index, image_li in enumerate(image_lis):
      if index == self.k:
        break
      self.download_image(engine, image_li['data-objurl'], s, _class, index, folder_name)

  def get_topk_from_google(self, engine, html, s, _class, forder_name):
    image_lis = html.find_all('div', class_='isv-r PNCib MSM1fd BUooTd')
    for index, image_li in enumerate(image_lis):
      if index == self.k:
        break
      a = image_li.find('a')
      img = a.find('img')
      if img.get('src', None):
        if re.search(r'data:image/(.*);', img['src']):
          self.decode_image(engine, img['src'], s, _class, index, forder_name)
        else:
          self.download_image(engine, img['src'], s, _class, index, forder_name)
  
  def get_topk_from_bing(self, engine, html, s, _class, folder_name):
    image_as = html.find_all('a', class_='iusc')
    for index, image_a in enumerate(image_as):
      if index == self.k:
        break
      img = image_a.find('img')
      if img.get('src', None):
        if re.search(r'data:image/(.*);', img['src']):
          self.decode_image(engine, img['src'], s, _class, index, folder_name)
        else:
          self.download_image(engine, img['src'], s, _class, index, folder_name)

  def get_topk_from_yahoo(self, engine, html, s, _class, folder_name):
    image_lis = html.find_all('li', class_='ld')
    for index, image_li in enumerate(image_lis):
      if index == self.k:
        break
      img = image_li.find('img')
      if img.get('src', None):
        if re.search(r'data:image/(.*);', img['src']):
          self.decode_image(engine, img['src'], s, _class, index, folder_name)
        else:
          self.download_image(engine, img['src'], s, _class, index, folder_name)

  def get_topk_from_aol(self):
    pass

  def get_topk_from_duckduckgo(self, engine, html, s, _class, folder_name):
    image_divs = html.find_all('div', class_='tile tile--img has-detail')
    for index, image_div in enumerate(image_divs):
      if index == self.k:
        break
      img = image_div.find('img')
      if img.get('src', None):
        if re.search(r'data:image/(.*);', img['src']):
          self.decode_image(engine, img['src'], s, _class, index, folder_name)
        else:
          http_position = img['src'].find('https')
          self.download_image(engine, unquote(img['src'][http_position:]), s, _class, index, folder_name)

  def download_image(self, engine, image_url, s, _class, index, forder_name):
    image_folder = '{}{}/{}/'.format(self.data_path, engine, forder_name)
    if not os.path.exists(image_folder):
      os.makedirs(image_folder)
    try:
      response = requests.get(image_url)
      content_type = response.headers['content-type']
      slash_postition = content_type.rfind('/')
      img_type = content_type[slash_postition+1:]
      with open(image_folder + '{}.{}'.format(index, img_type), 'wb') as file:
        file.write(response.content)
    except:
      print(image_url)

  def decode_image(self, engine, image_src, s, _class, index, forder_name):
    image_folder = '{}{}/{}/'.format(self.data_path, engine, forder_name)
    if not os.path.exists(image_folder):
      os.makedirs(image_folder)
    try:
      img_type = re.search(r'data:image/(.*);', image_src).groups()[0]
      image_src = re.sub(r'data:image/(.*);base64,', '', image_src)
      with open(image_folder + '{}.{}'.format(index, img_type), 'wb') as file:
        file.write(base64.b64decode(image_src))
    except:
      print(image_src)

  def get_topk_from_engine(self, s, _class, folder_name=None):
    if not folder_name:
      folder_name = re.sub(r'[\"|\/|\?|\*|\:|\||\\|\<|\>]', ' ', s)
    if not self.driver:
      self.driver = self.start_brower()
    for engine in self.engine:
      search_pattern = self.search_patterns[engine]
      search_url = search_pattern.format(quote(s + ' ' + _class, 'utf-8'), self.k)
      while True:
        try:
          self.driver.get(search_url)
          break
        except TimeoutException as e:
          pass
      html = BeautifulSoup(self.driver.page_source, 'html.parser')
      self.searchEngine[engine](engine, html, s, _class, folder_name)

if __name__ == "__main__":
  imageEngine = ImageSearchEngine(engine=['duckduckgo'])
  imageEngine.get_topk_from_engine('博美犬', '动物')