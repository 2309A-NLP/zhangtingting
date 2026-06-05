import fitz, json, os, re, random

random.seed(42)
PDF = '/mnt/d/Desktop/NLP-RAG-11/data/raw/2020-03-26.pdf'
OUT = 'data/processed'

# ================ PAGE RANGES FROM TOC ================
# Page numbers from the actual TOC (1-indexed)
SECTIONS_BY_PAGE = [
    ('glossary',             '释义',             4, 6),       # pages 4-6
    ('company_info',         '公司基本情况',      8, 9),       # pages 8-9 (+ page 10 is auditors)
    ('financial_summary',    '财务概要',          11, 14),     # pages 10-14
    ('leadership',           '领导致辞',          18, 22),     # pages 17-22
    ('discussion_outlook',   '环境与展望',        24, 24),     # page 23
    ('discussion_financial', '财务报表分析',      26, 48),     # pages 25-47
    ('discussion_business',  '业务综述',          49, 72),     # pages 48-71
    ('discussion_risk',      '风险管理',          73, 93),     # pages 72-92
    ('discussion_capital',   '资本管理',          94, 96),     # pages 93-95
    ('shareholding',         '股本变动及股东情况',  97, 111),    # pages 96-110
    ('management',           '董监事高管',        112, 125),   # pages 111-124
    ('governance',           '公司治理',          126, 150),   # pages 125-149
    ('board_report',         '董事会报告',        151, 159),   # pages 150-158
    ('supervisory_report',   '监事会报告',        160, 163),   # pages 159-162
    ('related_transactions', '关联交易',          164, 171),   # pages 163-170
    ('important_matters',    '重要事项',          172, 421),   # pages 171-421 (rest)
]

# ================ TRAIN/TEST SPLIT ================
# Hold out entire sections as test set (never seen during training)
# This gives a realistic RAG evaluation: test on unseen topics
TEST_SECTIONS = {'glossary', 'leadership', 'supervisory_report'}

def extract_page(pdf_path, page_num):
    """Extract clean lines from a single page."""
    doc = fitz.open(pdf_path)
    page = doc[page_num - 1]  # 0-indexed
    blocks = page.get_text('blocks')
    lines = []
    for b in blocks:
        t = b[4].strip()
        if not t:
            continue
        for l in t.split('\n'):
            l = l.strip()
            if l and not re.match(r'^\d{1,4}$', l):
                lines.append(l)
    doc.close()
    return lines

def is_table_line(l):
    d = sum(1 for c in l if c.isdigit())
    ns = len(l) - l.count(' ')
    return ns > 0 and d / ns > 0.5

def is_toc_line(l):
    dots = sum(1 for c in l if c in '.·•\u00b7\u30fb')
    return dots > 5 and len(l) > 30

def get_paras(lines):
    """Extract clean paragraphs from lines."""
    paras = []
    cur = []
    for l in lines:
        l = l.strip()
        if not l:
            if cur:
                t = ''.join(cur)
                if len(t) > 80:
                    paras.append(t)
                cur = []
            continue
        if is_toc_line(l) or is_table_line(l):
            if cur:
                t = ''.join(cur)
                if len(t) > 80:
                    paras.append(t)
                cur = []
            continue
        cur.append(l)
    if cur:
        t = ''.join(cur)
        if len(t) > 80:
            paras.append(t)
    return [p for p in paras if sum(1 for c in p if '\u4e00' <= c <= '\u9fff') >= 60]

def find_subsection_headers(lines):
    """Find subsection headers within a section."""
    pat = re.compile(r'^(\d+\.\d+\.\d+|\d+\.\d+\.\d+\.\d+)\s+')
    subs = []
    for i, l in enumerate(lines):
        m = pat.match(l.strip())
        if m and len(l) <= 50 and not is_toc_line(l):
            subs.append((l.strip(), i))
    return subs

def extract_glossary(pages_to_extract):
    """Extract glossary term-definition pairs from specified pages."""
    all_lines = []
    for pn in pages_to_extract:
        all_lines.extend(extract_page(PDF, pn))
    
    entries = []
    i = 0
    while i < len(all_lines):
        l = all_lines[i]
        if '\u201c' in l or '"' in l:
            tm = re.search(r'[\u201c"]([^\u201c"]+)[\u201d"]', l)
            if tm:
                term = tm.group(1).replace(' ', '').replace('\u3000', '').strip()
                j = i + 1
                def_parts = []
                while j < len(all_lines):
                    nl = all_lines[j]
                    if ('\u201c' in nl or '"' in nl) and '指' not in nl[:10] and len(nl.strip()) < 30:
                        break
                    if nl.strip() in ['指', '指'] or nl.strip() == '指':
                        j += 1
                        continue
                    def_parts.append(nl.replace(' ', '').replace('\u3000', ''))
                    j += 1
                definition = ''.join(def_parts)
                if len(definition) >= 20:
                    entries.append((f'{term}: {definition}', term))
                i = j
                continue
        i += 1
    return entries

def gen_queries(text, sect_key, sub_title=''):
    qs = set()
    # Section-level
    def cn_lookup(key):
        cn_map = {
            'glossary': '释义', 'company_info': '公司基本情况',
            'financial_summary': '财务概要', 'leadership': '领导致辞',
            'discussion_outlook': '环境与展望',
            'discussion_financial': '财务报表分析',
            'discussion_business': '业务综述', 'discussion_risk': '风险管理',
            'discussion_capital': '资本管理', 'shareholding': '股本变动',
            'management': '董监事高管', 'governance': '公司治理',
            'board_report': '董事会报告', 'supervisory_report': '监事会报告',
            'related_transactions': '关联交易', 'important_matters': '重要事项',
        }
        return cn_map.get(key, '')
    cn = cn_lookup(sect_key)
    if cn:
        qs.add(f'邮储银行{cn}介绍')
        qs.add(f'邮储银行{cn}主要有哪些内容？')
    if sub_title:
        st = re.sub(r'^[\d\.\s]+', '', sub_title).strip()
        if st:
            qs.add(f'{st}是什么？')
            qs.add(f'邮储银行{st}情况如何？')
    terms = re.findall(r'[\u4e00-\u9fff]{2,8}(?:业务|风险|贷款|存款|管理|体系|客户|服务|产品|收入|利润|资产|市场|改革|转型|战略|发展)', text[:300])
    for t in list(set(terms))[:2]:
        qs.add(f'邮储银行{t}情况')
    stats = re.findall(r'([\u4e00-\u9fff]{3,12})(?:达|为|增长|下降|增加|减少)([\d,]+)', text[:400])
    for item, _ in stats[:2]:
        qs.add(f'邮储银行{item}是多少？')
    if sect_key == 'glossary':
        tm = re.search(r'[\u201c"]([^\u201c"]+)[\u201d"]', text[:80])
        if tm:
            tname = tm.group(1).replace(' ', '').strip()
            qs.add(f'{tname}是什么意思？')
            qs.add(f'什么是{tname}？')
    final = []
    for q in qs:
        q = q.replace('邮储银行邮储银行', '邮储银行')
        if 6 <= len(q) <= 60:
            final.append(q)
    return final[:5]

def generate_all_datasets(out_dir=OUT):
    print('=' * 60)
    print('Generating Training Data from PSBC 2019 Annual Report')
    print('=' * 60)
    
    print('[1] Extracting glossary terms...')
    glossary_entries = extract_glossary(range(4, 7))  # pages 4-6
    print(f'  {len(glossary_entries)} term-definition entries')
    
    print('[2] Extracting content from each section...')
    sect_paras = {
        'glossary': glossary_entries
    }
    
    for key, cn_name, start_page, end_page in SECTIONS_BY_PAGE:
        if key == 'glossary':
            continue
        print(f'  {key:30s} pages {start_page}-{end_page}...', end=' ')
        
        # Extract all lines from these pages
        all_lines = []
        for pn in range(start_page, end_page + 1):
            all_lines.extend(extract_page(PDF, pn))
        
        subs = find_subsection_headers(all_lines)
        
        if subs:
            count = 0
            for si, (st, sl) in enumerate(subs):
                se = subs[si + 1][1] if si + 1 < len(subs) else len(all_lines)
                paras = get_paras(all_lines[sl:se])
                sub_clean = re.sub(r'^[\d\.\s]+', '', st).strip()
                if key not in sect_paras:
                    sect_paras[key] = []
                for p in paras:
                    sect_paras[key].append((p, sub_clean))
                    count += 1
            print(f'{count} paragraphs from {len(subs)} subsections')
        else:
            paras = get_paras(all_lines)
            if key not in sect_paras:
                sect_paras[key] = []
            for p in paras:
                sect_paras[key].append((p, ''))
            print(f'{len(paras)} paragraphs')
    
    total = sum(len(v) for v in sect_paras.values())
    print(f'  Total: {total} text blocks across {len(sect_paras)} sections')
    
    print('[3] Generating queries...')
    pos_pairs = []
    meta = []
    for sk, paras in sect_paras.items():
        for pt, sub in paras:
            qs = gen_queries(pt, sk, sub)
            for q in qs:
                pos_pairs.append((q, pt, 0.9))
                meta.append({'q': q, 'd': pt, 's': sk, 'sub': sub})
    print(f'  {len(pos_pairs)} query-doc pairs')
    
    # ==================== TRAIN/TEST SPLIT ====================
    train_pairs = []
    train_meta = []
    test_pairs = []
    test_meta = []
    for m, pair in zip(meta, pos_pairs):
        if m['s'] in TEST_SECTIONS:
            test_pairs.append(pair)
            test_meta.append(m)
        else:
            train_pairs.append(pair)
            train_meta.append(m)
    print(f'  Train: {len(train_pairs)} pairs ({len(set(m["s"] for m in train_meta))} sections)')
    print(f'  Test:  {len(test_pairs)} pairs ({len(set(m["s"] for m in test_meta))} sections)')
    
    print('[4] Building triplets (train only)...')
    groups = {
        'biz': {'discussion_business'},
        'risk': {'discussion_risk'},
        'fin': {'discussion_financial', 'financial_summary', 'discussion_capital', 'discussion_outlook'},
        'gov': {'governance', 'management', 'board_report', 'supervisory_report'},
        'co': {'company_info', 'leadership', 'shareholding'},
        'gl': {'glossary'},
        'other': {'related_transactions', 'important_matters'},
    }
    rev_g = {k: g for g, ks in groups.items() for k in ks}
    idx = {}
    for m in train_meta:
        idx.setdefault(m['s'], []).append(m['d'])
    trips = []
    for m in train_meta:
        q, pos, s = m['q'], m['d'], m['s']
        my_g = rev_g.get(s)
        neg = None
        if my_g:
            for a in [k for k in groups[my_g] if k != s]:
                if a in idx and idx[a]:
                    neg = random.choice(idx[a])
                    break
        if not neg:
            others = [k for k in idx if k != s]
            if others:
                neg = random.choice(idx[random.choice(others)])
        if neg and neg != pos:
            trips.append((q, pos, neg))
    print(f'  {len(trips)} triplets')
    
    print('[5] Building cosine pairs...')
    cos = list(train_pairs)
    # Only use train sections for cosine pairs
    train_sect_paras = {k: v for k, v in sect_paras.items() if k not in TEST_SECTIONS}
    for sk, paras in train_sect_paras.items():
        pl = [(p, sub) for p, sub in paras if sub]
        random.shuffle(pl)
        for i in range(min(len(pl), 50)):
            for j in range(i + 1, min(len(pl), i + 5)):
                if pl[i][1] != pl[j][1]:
                    cos.append((pl[i][0][:60], pl[j][0], 0.5))
    keys = list(train_sect_paras.keys())
    for _ in range(400):
        k1, k2 = random.sample(keys, 2)
        if train_sect_paras[k1] and train_sect_paras[k2]:
            p1 = random.choice(train_sect_paras[k1])
            p2 = random.choice(train_sect_paras[k2])
            cos.append((p1[0][:60], p2[0], 0.1))
    print(f'  {len(cos)} cosine pairs')
    
    print(f'[6] Saving to {out_dir}/...')
    os.makedirs(out_dir, exist_ok=True)
    def sv(obj, name):
        p = os.path.join(out_dir, name)
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        return len(obj)
    
    r = {
        'train_positive_pairs': sv(train_pairs, 'psbc_train_positive_pairs.json'),
        'test_positive_pairs': sv(test_pairs, 'psbc_test_positive_pairs.json'),
        'triplets': sv(trips, 'psbc_triplets.json'),
        'cosine_pairs': sv(cos, 'psbc_cosine_pairs.json'),
        'combined': sv(train_pairs + [(a, p, n, 'triplet') for a, p, n in trips], 'psbc_combined_train.json'),
    }
    sv({'source': 'PSBC 2019 Annual Report', 'total_pages': 421,
        'paragraphs': total, 'sections': len(sect_paras),
        'train_sections': len(sect_paras) - len(TEST_SECTIONS),
        'test_sections': len(TEST_SECTIONS),
        'test_section_names': sorted(TEST_SECTIONS),
        'datasets': r}, 'psbc_dataset_metadata.json')
    
    print('\n' + '=' * 60)
    print('Complete!')
    for k, v in r.items():
        print(f'  {k}: {v}')
    print('=' * 60)
    return r

# ================ LEGACY CONSTANTS (for legal/medical domain eval) ================
LEGAL_DOCUMENTS = []
LEGAL_QUERIES = []
QUERY_DOC_MAP = {}
MEDICAL_DOCUMENTS = []
MEDICAL_QUERIES = []
MEDICAL_QUERY_DOC_MAP = {}

if __name__ == '__main__':
    import sys
    generate_all_datasets(sys.argv[1] if len(sys.argv) > 1 else OUT)