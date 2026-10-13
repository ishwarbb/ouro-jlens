"""Validate local report navigation and render its portable PDF."""
from html.parser import HTMLParser
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlparse

ROOT=Path(__file__).resolve().parent
REPORT=ROOT/'results/run-2'

class Links(HTMLParser):
    def __init__(self):
        super().__init__();self.links=[];self.ids=set()
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if 'id' in attrs:self.ids.add(attrs['id'])
        if tag=='a' and 'href' in attrs:self.links.append(attrs['href'])

def main():
    from playwright.sync_api import sync_playwright
    errors=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
        page=browser.new_page(viewport={'width':1280,'height':960},device_scale_factor=1)
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto((REPORT/'index.html').as_uri(),wait_until='networkidle')
        assert page.title().startswith('Run 2')
        assert page.locator('h1').inner_text()=='Run 2'
        assert page.locator('.stat strong').all_text_contents()==['260','68','85','2']
        page.screenshot(path=str(REPORT/'preview.png'),full_page=True)
        page.screenshot(path=str(REPORT/'preview-top.png'))
        page.pdf(path=str(REPORT/'Run 2.pdf'),format='A4',print_background=True,prefer_css_page_size=True)
        page.set_viewport_size({'width':390,'height':844})
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), 'Mobile horizontal overflow'
        plan=ROOT/'results/run-3-plan'
        page.goto((plan/'index.html').as_uri(),wait_until='networkidle')
        assert page.title().startswith('Run 3')
        assert page.locator('details').count()==24
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), 'Plan mobile horizontal overflow'
        page.set_viewport_size({'width':1280,'height':960})
        page.screenshot(path=str(plan/'preview-top.png'))
        page.pdf(path=str(plan/'Run 3 plan.pdf'),format='A4',print_background=True,prefer_css_page_size=True)
        browser.close()
    parser=Links();parser.feed((REPORT/'index.html').read_text());checked=0
    for href in parser.links:
        url=urlparse(href)
        if url.scheme or not url.path:continue
        path=(REPORT/unquote(url.path)).resolve()
        assert path.is_file(),href
        if url.fragment:
            target=Links();target.feed(path.read_text());assert unquote(url.fragment) in target.ids,href
        checked+=1
    for relative,sha in json.loads((REPORT/'manifest.json').read_text()).items():
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==sha,relative
    pdf=(REPORT/'Run 2.pdf').read_bytes();assert pdf.startswith(b'%PDF-') and b'%%EOF' in pdf
    assert not errors,errors
    plan_parser=Links();plan_parser.feed((plan/'index.html').read_text())
    for href in plan_parser.links:
        url=urlparse(href)
        if url.scheme or not url.path:continue
        assert (plan/unquote(url.path)).resolve().is_file(),href
    for relative,sha in json.loads((plan/'manifest.json').read_text()).items():
        assert hashlib.sha256((plan/relative).read_bytes()).hexdigest()==sha,relative
    plan_protocol=json.loads((plan/'protocol.json').read_text())
    templates=json.loads((plan/'prompt-templates.json').read_text())
    calibration=json.loads((plan/'calibration-templates.json').read_text())
    assert plan_protocol['status']=='planned_not_executed'
    assert len(templates)==24
    assert sum(len(t['seeds']) for r in templates for t in r['templates'])==576
    assert sum(len(t['seeds']) for t in calibration)==48
    plan_result=dict(status='passed',mobile_overflow=False,prompt_records=24,condition_templates=96,
                     planned_observational_generations=648,model_inference_launched=False,
                     pdf_bytes=(plan/'Run 3 plan.pdf').stat().st_size)
    (plan/'report-verification.json').write_text(json.dumps(plan_result,indent=2)+'\n')
    result=dict(status='passed',local_links_checked=checked,browser_errors=errors,mobile_overflow=False,pdf_bytes=len(pdf),pdf_pages=len(re.findall(rb'/Type\s*/Page\b',pdf)),pdf_sha256=hashlib.sha256(pdf).hexdigest())
    (REPORT/'report-verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
