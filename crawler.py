#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cultural Atlas 深度禁忌内容爬虫（整段提取版）
"""
import requests
from bs4 import BeautifulSoup
import re
import json
import csv
import time
from urllib.parse import urljoin
from collections import defaultdict

class CulturalAtlasDeepCrawler:
    def __init__(self):
        self.base_url = "https://culturalatlas.sbs.com.au"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        })
        self.subcategories = [
            'greetings', 'religion', 'family', 'naming', 'dates-of-significance',
            'etiquette', 'do-s-and-don-ts', 'communication', 'other-considerations', 'business-culture'
        ]
        self.taboo_keywords = [
            r"\bdon't\b", r"\bdon\'t\b", r"\bavoid\b", r"\bnever\b", r"\bshouldn't\b", r"\bshouldn\'t\b", r"\bwon't\b", r"\bwon\'t\b",
            r"\bcan't\b", r"\bcan\'t\b", r"\bmustn't\b", r"\bmustn\'t\b", r"\bforbidden\b", r"\bprohibited\b", r"\bnot allowed\b", r"\bnot acceptable\b",
            r"\billegal\b", r"\bunlawful\b", r"\bdo not\b", r"\bdo NOT\b", r"\bwarning\b", r"\bcaution\b", r"\bbeware\b", r"\bcareful\b",
            r"\bwatch out\b", r"\bavoid doing\b", r"\bnot recommended\b", r"\boffensive\b", r"\binappropriate\b", r"\btaboo\b", r"\bdisrespectful\b",
            r"\binsensitive\b", r"\brude\b", r"\bimpolite\b", r"\bunacceptable\b", r"\bunsuitable\b", r"\binsulting\b", r"\boffend\b", r"\boffends\b",
            r"\brefrain from\b", r"\bstay away from\b", r"\bkeep away from\b", r"\bmake sure not to\b", r"\bbe careful not to\b", r"\bensure you don't\b",
            r"\bit's best not to\b", r"\byou should not\b", r"\btry not to\b", r"\bhighly discouraged\b", r"\bstrongly advised against\b", r"\bnot wise to\b",
            r"\bbad idea to\b", r"\bunwise to\b"
        ]
        self.taboo_pattern = re.compile('|'.join(self.taboo_keywords), re.IGNORECASE)
        self.results = []
        self.culture_stats = defaultdict(int)
        self.category_stats = defaultdict(int)

    def get_culture_names(self):
        print("🔍 获取文化列表...")
        try:
            response = self.session.get(f"{self.base_url}/countries", timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            culture_names = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                if href.endswith('-culture') and href.startswith('/'):
                    culture_name = href[1:]
                    culture_names.append(culture_name)
            culture_names = list(set(culture_names))
            culture_names.sort()
            print(f"✅ 找到 {len(culture_names)} 个文化")
            return culture_names
        except Exception as e:
            print(f"❌ 获取文化列表失败: {e}")
            return []

    def extract_taboo_paragraphs(self, text):
        """提取包含禁忌关键词的整段"""
        paragraphs = re.split(r'\n{2,}|\r{2,}', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip() and len(p.strip()) > 20]
        taboo_paragraphs = []
        for para in paragraphs:
            matches = list(self.taboo_pattern.finditer(para))
            if matches:
                clean_para = ' '.join(para.split())
                taboo_paragraphs.append({
                    'paragraph': clean_para,
                    'keywords': [m.group() for m in matches],
                    'length': len(clean_para)
                })
        return taboo_paragraphs

    def crawl_subcategory_page(self, culture_name, subcategory):
        culture_base = culture_name.replace('-culture', '')
        url = f"{self.base_url}/{culture_name}/{culture_base}-{subcategory}"
        try:
            response = self.session.get(url, timeout=10)
            if response.status_code == 404:
                alt_url = f"{self.base_url}/{culture_name}/{culture_name}-{subcategory}"
                response = self.session.get(alt_url, timeout=10)
                if response.status_code == 200:
                    url = alt_url
            if response.status_code != 200:
                return []
            soup = BeautifulSoup(response.content, 'html.parser')
            title_element = soup.find('title')
            page_title = title_element.get_text().strip() if title_element else f"{culture_name} - {subcategory}"
            main_content = (soup.find('main') or 
                           soup.find('article') or 
                           soup.find('div', class_=re.compile(r'content|main|body', re.I)))
            content_area = main_content if main_content else soup
            for unwanted in content_area(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                unwanted.decompose()
            text_content = content_area.get_text()
            taboo_paragraphs = self.extract_taboo_paragraphs(text_content)
            if taboo_paragraphs:
                result = {
                    'culture': culture_name.replace('-culture', '').replace('-', ' ').title(),
                    'category': subcategory,
                    'url': url,
                    'title': page_title,
                    'taboo_paragraphs': taboo_paragraphs,
                    'count': len(taboo_paragraphs)
                }
                self.results.append(result)
                self.culture_stats[culture_name] += len(taboo_paragraphs)
                self.category_stats[subcategory] += len(taboo_paragraphs)
                print(f"    ✅ {subcategory}: 找到 {len(taboo_paragraphs)} 条禁忌段落")
                for i, para in enumerate(taboo_paragraphs[:2]):
                    print(f"       {i+1}. {para['paragraph'][:80]}...")
                return taboo_paragraphs
            return []
        except Exception as e:
            return []

    def crawl_culture_deep(self, culture_name):
        culture_display = culture_name.replace('-culture', '').replace('-', ' ').title()
        print(f"\n🌍 爬取文化: {culture_display}")
        total_found = 0
        for subcategory in self.subcategories:
            taboo_content = self.crawl_subcategory_page(culture_name, subcategory)
            total_found += len(taboo_content)
            time.sleep(0.3)
        if total_found > 0:
            print(f"  📊 总计找到 {total_found} 条禁忌内容")
        else:
            print(f"  ➖ 未找到禁忌内容")
        return total_found

    def crawl_all_cultures(self, max_cultures=None):
        culture_names = self.get_culture_names()
        if not culture_names:
            print("❌ 未找到任何文化页面")
            return []
        if max_cultures:
            culture_names = culture_names[:max_cultures]
            print(f"🔢 限制爬取前 {max_cultures} 个文化")
        print(f"\n🚀 开始深度爬取 {len(culture_names)} 个文化的详细内容...")
        print("=" * 80)
        total_cultures_with_content = 0
        total_sentences = 0
        for i, culture_name in enumerate(culture_names, 1):
            print(f"\n[{i}/{len(culture_names)}] ", end="")
            found_count = self.crawl_culture_deep(culture_name)
            if found_count > 0:
                total_cultures_with_content += 1
                total_sentences += found_count
        print(f"\n{'='*80}")
        print(f"🎉 爬取完成!")
        print(f"📈 有内容的文化数: {total_cultures_with_content}/{len(culture_names)}")
        print(f"📝 总禁忌内容数: {total_sentences}")
        return self.results

    def save_detailed_results(self, json_file="detailed_cultural_taboos_paragraph.json", csv_file="detailed_cultural_taboos_paragraph.csv"):
        if not self.results:
            print("❌ 没有结果需要保存")
            return
        output_data = {
            'summary': {
                'total_cultures': len(set(r['culture'] for r in self.results)),
                'total_pages': len(self.results),
                'total_paragraphs': sum(r['count'] for r in self.results),
                'crawl_date': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'culture_stats': dict(self.culture_stats),
            'category_stats': dict(self.category_stats),
            'detailed_results': self.results
        }
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"✅ 详细JSON结果已保存: {json_file}")
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['文化', '类别', 'URL', '页面标题', '禁忌段落', '关键词', '段落长度'])
            for page in self.results:
                for para in page['taboo_paragraphs']:
                    writer.writerow([
                        page['culture'],
                        page['category'],
                        page['url'],
                        page['title'],
                        para['paragraph'],
                        ', '.join(para['keywords']),
                        para['length']
                    ])
        print(f"✅ 详细CSV结果已保存: {csv_file}")

    def print_comprehensive_summary(self):
        if not self.results:
            print("❌ 未找到任何禁忌内容")
            return
        total_paragraphs = sum(r['count'] for r in self.results)
        total_cultures = len(set(r['culture'] for r in self.results))
        print(f"\n{'='*80}")
        print(f"📊 Cultural Atlas 深度禁忌内容分析报告（整段版）")
        print(f"{'='*80}")
        print(f"🌍 涉及文化数量: {total_cultures}")
        print(f"📄 有内容的页面数: {len(self.results)}")
        print(f"📝 总禁忌段落数: {total_paragraphs}")
        print(f"📈 平均每文化: {total_paragraphs/total_cultures:.1f} 条")
        print(f"\n📋 按内容类别统计:")
        sorted_categories = sorted(self.category_stats.items(), key=lambda x: x[1], reverse=True)
        for category, count in sorted_categories:
            print(f"  📌 {category:<25} {count:3d} 条")
        print(f"\n🌍 禁忌内容最多的文化 (Top 10):")
        sorted_cultures = sorted(self.culture_stats.items(), key=lambda x: x[1], reverse=True)[:10]
        for i, (culture, count) in enumerate(sorted_cultures, 1):
            display_name = culture.replace('-culture', '').replace('-', ' ').title()
            print(f"  {i:2d}. {display_name:<25} {count:3d} 条")
        keyword_count = defaultdict(int)
        for result in self.results:
            for para in result['taboo_paragraphs']:
                for keyword in para['keywords']:
                    keyword_count[keyword.lower()] += 1
        print(f"\n🔤 最常见的禁忌关键词 (Top 15):")
        sorted_keywords = sorted(keyword_count.items(), key=lambda x: x[1], reverse=True)[:15]
        for i, (keyword, count) in enumerate(sorted_keywords, 1):
            print(f"  {i:2d}. {keyword:<20} {count:3d} 次")
        print(f"\n📄 禁忌内容精选示例:")
        example_count = 0
        for result in self.results[:5]:
            if example_count >= 10:
                break
            print(f"\n🌍 【{result['culture']}】- {result['category']}")
            for para in result['taboo_paragraphs'][:2]:
                if example_count >= 10:
                    break
                print(f"   💬 {para['paragraph'][:150]}...")
                print(f"      🏷️  关键词: {', '.join(para['keywords'])}")
                example_count += 1

def main():
    print("🌍 Cultural Atlas 深度禁忌内容爬虫（整段提取版）")
    print("📊 基于真实网站结构的完整版本")
    print("🎯 爬取路径: 国家列表 → 文化主页 → 具体内容子页面")
    print("="*80)
    crawler = CulturalAtlasDeepCrawler()
    try:
        results = crawler.crawl_all_cultures()
        crawler.print_comprehensive_summary()
        if results:
            crawler.save_detailed_results()
            print(f"\n✨ 任务完成！生成的文件:")
            print(f"   📁 detailed_cultural_taboos_paragraph.json - 包含统计信息的完整JSON数据")
            print(f"   📊 detailed_cultural_taboos_paragraph.csv - 详细CSV表格数据")
    except KeyboardInterrupt:
        print(f"\n⚠️  用户中断了爬取过程")
        if crawler.results:
            print("💾 保存已获取的部分结果...")
            crawler.save_detailed_results()
    except Exception as e:
        print(f"\n❌ 爬取过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        if crawler.results:
            print("💾 保存已获取的部分结果...")
            crawler.save_detailed_results()

if __name__ == "__main__":
    main()
