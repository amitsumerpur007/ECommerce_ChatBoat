import pandas as pd
from pathlib import Path
from langchain_core.documents import Document




def dataconverter():
    data_path = Path('E:/Projects/GenAI/Ecommerce_chatboat/data/flipkart_product_review.csv')
    data = pd.read_csv(data_path)
    print(data.head())

    data = data[['product_title', 'review']]
    data_dict = data.to_dict('records')
    data_dict = [{'product_name': d['product_title'], 'review': d['review']} for d in data_dict]

    docs = []
    for entry in data_dict:

        
        metadata = {"product_name": entry['product_name']}
        doc = Document(page_content=entry['review'],metadata=metadata)
        docs.append(doc)
    return docs