from icrawler.builtin import GoogleImageCrawler

fruits = ['apple', 'bannana', 'carrot', 'mango', 'onion', 'potato', 'rice', 'tomoto', 'wheat']

for fruit in fruits:
    crawler = GoogleImageCrawler(storage={'root_dir': f'dataset/{fruit}'})
    crawler.crawl(keyword=fruit, max_num=200)