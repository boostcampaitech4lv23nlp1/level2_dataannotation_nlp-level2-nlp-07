import wikipediaapi
from kiwipiepy import Kiwi
import pandas as pd
major = major = ['국어국문학', '기독교학', '독어독문학', '불어불문학', '사학', '스포츠학', '영어영문학', '일어일문학', '중어중문학', '철학', '건축학', '기계공학', '물리학', '산업공학', '소프트웨어학', '수학', '신소재공학', '의생명시스템공학', '전자공학','통계학', '정보통신전자공학', '컴퓨터학', '화학공학', '화학', '문예창작학', '경영학', '경제학', '국제무역학', '글로벌통상학', '금융경제학', '벤처경영학', '벤처중소기업학', '회계학', '사회복지학부','보안학', '글로벌미디어학', '미디어경영학', '언론홍보학', '정보사회학', '정치외교학', '평생교육학', '행정학', '국제법무학', '법학']

if __name__ == '__main__':
    kiwi = Kiwi()
    wiki_wiki = wikipediaapi.Wikipedia('ko')
    cnt = 0
    total_len = 0
    major = ['서양 철학']
    for i in major:
        page_py = wiki_wiki.page(i)
        if page_py.exists():
            tmp = page_py.text
            pls = kiwi.split_into_sents(tmp)
            sentence = [x.text for x in pls]
            sent = pd.Series(sentence)
            print(f'{i} sentence has {len(sent)} lens')
            sent.to_csv(f'output/{i}.csv',index=False)
            total_len += len(sent)
        else:
            cnt += 1
            
    print(f'total process : {len(major)}')
    print(f'done process : {len(major)- cnt}')
    print(f'missing pages : {cnt}')
    print(f'total_len : {total_len}, 엔빵_len : {total_len//5}')    
