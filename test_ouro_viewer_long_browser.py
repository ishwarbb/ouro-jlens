"""Live acceptance check for long output, cached window navigation and sharing.

Run with a populated viewer and Playwright; performs real model work.
"""
import argparse
import json
from pathlib import Path
import time


def main():
    from playwright.sync_api import sync_playwright
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',default='http://127.0.0.1:8769')
    parser.add_argument('--output',type=Path,default=Path('/tmp/ouro-long-browser-report.json'))
    args=parser.parse_args();report=dict(checks=[],browser_errors=[])
    def passed(name,**details):
        report['checks'].append(dict(name=name,**details));print('PASS:',name,details,flush=True)
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
        context=browser.new_context(viewport=dict(width=1440,height=1000),accept_downloads=True)
        page=context.new_page();page.on('pageerror',lambda e:report['browser_errors'].append(str(e)))
        page.goto(args.url)
        page.wait_for_function('S.run && document.getElementById("run-loader").hidden')
        assert page.locator('#max-tokens').input_value()=='16819'
        page.locator('[data-run="52222d748155"]').click()
        page.wait_for_function('S.run?.id === "52222d748155"')
        assert page.locator('#max-tokens').input_value()=='16819'
        assert page.evaluate('composerPayload().max_new_tokens')==16819
        passed('16,819 default survives opening an older 8-token run')

        page.locator('#new-run').click();page.locator('[data-mode="chat"]').click()
        page.locator('#prompt').fill('Reply with only the word: ready.')
        page.locator('#analyze-button').click()
        page.wait_for_function('!S.busy',timeout=180000)
        assert not page.locator('#error-banner').is_visible(),page.locator('#error-banner').inner_text()
        short=page.evaluate('({id:S.run.id,input:S.run.input,generation:S.run.generation})')
        assert short['input']['max_new_tokens']==16819
        assert short['generation']['stopped'] and not short['generation']['limit_reached']
        passed('16,819-token request stops naturally at end of turn',run_id=short['id'],generated=short['generation']['output_tokens'])

        page.locator('#new-run').click();page.locator('[data-mode="raw"]').click()
        page.locator('#prompt').fill('The waves rolled onto the shore.\n\n'*80)
        page.locator('#max-tokens').select_option('512');page.locator('#analyze-button').click()
        page.wait_for_function('!S.busy',timeout=360000)
        assert not page.locator('#error-banner').is_visible(),page.locator('#error-banner').inner_text()
        run=page.evaluate('({id:S.run.id,prompt_length:S.run.prompt_length,total:S.run.tokens.length,generation:S.run.generation})')
        assert run['generation']['output_tokens']>128,run
        assert run['prompt_length']>512 and run['total']>1024,run
        assert page.evaluate('S.run.layers.length')==192
        assert page.evaluate('S.window.start')==512
        assert page.locator('#transcript button').count()<=512
        assert page.locator('#continuation-preview p').inner_text()==run['generation']['answer_text'].strip()
        passed('generation exceeds 128; full output retained; initial window contains last prompt token',run_id=run['id'],prompt_length=run['prompt_length'],total=run['total'],generated=run['generation']['output_tokens'])

        last=(run['total']-1)//512*512
        page.locator('#token-window').select_option(str(last))
        page.wait_for_function('(start)=>S.window.start===start&&!S.windowBusy&&!S.busy',arg=last,timeout=180000)
        page.locator('#show-special').check()
        page.locator(f'#transcript [data-position="{run["total"]-1}"]').click()
        assert page.evaluate('cellItems(128,S.position)[0].id===S.layers.get(128).jlens.ids[S.position-S.window.start][0]')
        assert f'position {run["total"]-1} ' in page.locator('#cell-position').inner_text()
        page.locator('#pin-input').fill('sea');page.locator('#pin-form button').click()
        page.wait_for_function('S.traces.sea&&!S.pins.some(p=>p.loading)',timeout=180000)
        assert page.evaluate('S.traces.sea.position_start===S.window.start')
        assert page.evaluate('pinRank(128,S.position,S.pins[0]).rank===S.traces.sea.layers[128].jlens.ranks[S.position-S.window.start]')
        page.locator('#compute-inventory').click()
        page.wait_for_function('!S.busy && S.lens==="inventory"',timeout=180000)
        assert not page.locator('#error-banner').is_visible(),page.locator('#error-banner').inner_text()
        assert page.evaluate('cellItems(128,S.position,"inventory") !== null')
        passed('late-token J-lens readout, exact rank and sparse inventory use absolute positions',position=run['total']-1)

        shared=page.evaluate('viewState()');page.reload()
        page.wait_for_function('S.run && document.getElementById("run-loader").hidden&&!S.windowBusy&&!S.busy',timeout=180000)
        assert page.evaluate('viewState()')==shared
        assert page.evaluate('Boolean(S.traces.sea)')
        with page.expect_download() as pending:page.locator('#export-run').click()
        exported=json.loads(Path(pending.value.path()).read_text())
        assert len(exported['tokens'])==run['total']
        assert exported['readout_window']['start']==last
        assert exported['layers'][0]['position_start']==last
        passed('late-window share URL and export preserve full sequence and window offsets')

        page.locator('#token-window').select_option('0')
        page.wait_for_function('S.window.start===0&&!S.windowBusy&&!S.busy',timeout=180000)
        page.wait_for_function('!S.pins.some(p=>p.loading)',timeout=180000)
        assert page.evaluate('S.traces.sea.position_start')==0
        page.locator('#token-window').select_option(str(last))
        page.wait_for_function('(start)=>S.window.start===start&&!S.windowBusy&&!S.busy',arg=last,timeout=60000)
        assert page.evaluate('S.traces.sea.position_start')==last
        page.locator('#max-tokens').select_option('16819')
        for width in [320,390,768,1024]:
            page.set_viewport_size(dict(width=width,height=900))
            assert page.evaluate('document.body.scrollWidth<=innerWidth'),width
        passed('cached navigation restores matching traces; long-run controls fit mobile')
        assert not report['browser_errors'],report['browser_errors']
        browser.close()
    report['completed_at']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
