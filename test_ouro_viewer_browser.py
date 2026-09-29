"""Browser acceptance checks against a running, populated Ouro viewer.

Requires Playwright and Chromium in a separate environment. --live also runs
real model inference, an exact trace, sparse pursuit, and a causal edit.
"""
import argparse
import json
from pathlib import Path
import time

from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8769')
    parser.add_argument('--output', type=Path, default=Path('results/ouro-viewer/verification'))
    parser.add_argument('--live', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'url': args.url, 'live': args.live, 'checks': [], 'browser_errors': []}

    def passed(name, **details):
        report['checks'].append({'name': name, **details})
        print('PASS:', name, details, flush=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=['--no-sandbox'])
        context = browser.new_context(viewport={'width': 1440, 'height': 1080}, accept_downloads=True)
        page = context.new_page()
        page.on('pageerror', lambda e: report['browser_errors'].append(str(e)))
        page.set_default_timeout(30000)
        page.goto(args.url)
        page.wait_for_function('S.run && document.getElementById("run-loader").hidden')
        assert page.locator('#analysis').is_visible()
        assert page.evaluate('document.body.scrollWidth <= innerWidth')
        currency_id = page.evaluate('S.run.id')
        assert page.evaluate('S.run.layers.length') == 192
        assert 'Euro' in page.locator('#continuation-preview').inner_text()
        passed('recorded live run opens with all 192 sites', run_id=currency_id)

        page.locator('[data-lens="logit"]').click()
        assert page.locator('#readout-title').inner_text() == 'Logit lens readout'
        page.locator('[data-loop="0"]').click()
        page.locator('#workspace-only').uncheck()
        assert page.locator('#grid-viewport').get_attribute('aria-rowcount') == '192'
        page.locator('#position-scope').select_option('completion')
        assert page.evaluate('S.positions.every(p => S.run.tokens[p].segment === "completion")')
        before = page.evaluate('S.site')
        page.locator('#grid-viewport').focus()
        page.keyboard.press('ArrowDown')
        assert page.evaluate('S.site') == before + 1
        page.locator('#transcript [data-position="2"]').click()
        assert 'position 2 ' in page.locator('#cell-position').inner_text()
        # Return to a useful cell and verify its displayed value against the saved artifact.
        page.locator('#position-scope').select_option('all')
        page.locator('[data-loop="3"]').click()
        page.locator('#workspace-only').check()
        page.locator('[data-lens="jlens"]').click()
        prompt_last = page.evaluate('S.run.prompt_length - 1')
        page.locator(f'#transcript [data-position="{prompt_last}"]').click()
        expected = page.evaluate('tokenLabel(S.run.vocabulary[S.layers.get(S.site).jlens.ids[S.position][0]])')
        assert page.locator('#readout-list .readout-word').first.inner_text() == expected
        passed('lens, scope, loop, token selection and keyboard navigation agree with saved data')

        if args.live:
            page.locator('#pin-input').fill('Italy')
            page.locator('#pin-form button').click()
            page.wait_for_function('S.traces.italy && !S.pins.some(p => p.loading)', timeout=360000)
            assert page.evaluate('Object.keys(S.traces.italy.layers).length') == 192
            assert page.evaluate('S.traces.italy.layers[128].jlens.ranks.length === S.run.tokens.length')
            passed('exact full-vocabulary concept trace', concept='Italy')
            page.locator('#compute-inventory').click()
            page.wait_for_function('!S.busy && S.lens === "inventory"', timeout=360000)
            assert not page.locator('#error-banner').is_visible()
            assert page.locator('#readout-units').inner_text() == 'coefficient / ‖state‖'
            assert 'state energy reconstructed' in page.locator('#cell-reconstruction').inner_text()
            passed('real sparse cell inventory')

        page.locator('#inventory-k').select_option('16') if args.live else None
        page.locator('#inventory-threshold').select_option('0.01') if args.live else None
        page.locator('#show-special').check()
        page.locator('#zoom-grid').click()
        page.locator('#copy-link').click()
        shared_state = page.evaluate('viewState()')
        shared_url = page.url
        page.reload()
        page.wait_for_function('S.run && document.getElementById("run-loader").hidden')
        assert page.evaluate('viewState()') == shared_state
        if args.live:
            assert page.evaluate('Boolean(S.traces.italy) && Object.keys(S.cellInventory).length > 0')
            assert page.locator('#compute-inventory').is_hidden()
        with page.expect_download() as downloaded:
            page.locator('#export-run').click()
        path = args.output / 'export-smoke.json'
        downloaded.value.save_as(path)
        exported = json.loads(path.read_text())
        assert exported['id'] == currency_id and exported['viewer_state'] == shared_state
        if args.live:
            assert exported['exact_traces']['italy']['word'] == 'Italy'
            assert exported['inspected_inventories']
        path.unlink()
        passed('share URL, persistent extras and JSON download round trip', view=shared_state)

        page.locator('[data-run="research-animal"]').click()
        page.wait_for_function('S.run?.id === "research-animal"')
        page.locator('[data-lens="inventory"]').click()
        assert page.locator('#inventory-coverage').inner_text() == 'Full recorded grid'
        page.locator('#inventory-k').select_option('50')
        page.locator('#inventory-threshold').select_option('0')
        assert page.locator('#readout-list .readout-row').count() == 10
        page.locator('[data-loop="1"]').click()
        page.locator('#workspace-only').uncheck()
        page.locator('#grid-viewport').focus()
        for _ in range(48):
            page.keyboard.press('ArrowDown')
        assert page.evaluate('S.site') == 47
        assert 'not recorded' in page.locator('#readout-list').inner_text()
        passed('research inventory controls and explicitly missing loop-end target')
        page.locator('[data-run="research-sport"]').click()
        page.wait_for_function('S.run?.id === "research-sport"', timeout=60000)
        assert 'No final answer within this token budget.' in page.locator('#continuation-preview').inner_text()
        assert 'budget' in page.locator('#run-notices').inner_text()
        passed('capped thought-only output is not shown as a final answer')

        if args.live:
            page.locator('#new-run').click()
            page.locator('#prompt').fill('The capital of France is the city of')
            page.locator('#max-tokens').select_option('8')
            page.locator('#analyze-button').click()
            page.wait_for_function('!S.busy && S.run?.input.prompt === "The capital of France is the city of"', timeout=360000)
            assert not page.locator('#error-banner').is_visible()
            baseline = page.evaluate('({id:S.run.id, answer:S.run.generation.answer_text})')
            assert 'Paris' in baseline['answer']
            # Native final-site predictions must reproduce the generated sequence.
            alignment = page.evaluate('''() => {
              const l=S.layers.get(191),n=S.run.tokens.length;
              return Array.from({length:n-S.run.prompt_length},(_,i)=>S.run.prompt_length+i-1)
                .every(p=>l.logit.ids[p][0] === S.run.tokens[p+1].id);
            }''')
            assert alignment
            assert page.evaluate('[47,95,143,191].every(s => JSON.stringify(S.layers.get(s).jlens) === JSON.stringify(S.layers.get(s).logit))')
            passed('live greedy generation agrees with final-site readout; loop ends are identity transport', baseline=baseline)
            page.locator('#intervene-open').click()
            assert page.locator('#intervention-source').input_value() == 'France'
            assert page.locator('#intervention-destination').input_value() == 'China'
            page.locator('#intervention-form [type="submit"]').click()
            page.wait_for_function('!S.busy && S.run?.kind === "intervention"', timeout=360000)
            assert not page.locator('#error-banner').is_visible()
            edited = page.evaluate('({id:S.run.id, answer:S.run.generation.answer_text, baseline_id:S.run.baseline_id})')
            assert edited['baseline_id'] == baseline['id']
            assert 'Beijing' in edited['answer']
            assert 'Paris' in page.locator('#continuation-preview').inner_text()
            assert 'Beijing' in page.locator('#continuation-preview').inner_text()
            page.screenshot(path=str(args.output / 'intervention-desktop.png'), full_page=True)
            page.locator('#baseline-toggle').click()
            page.wait_for_function('(id) => S.run?.id === id', arg=baseline['id'])
            passed('France → China intervention changes Paris to Beijing and preserves baseline', edited=edited)

            # Submit via the UI, cancel before any completed artifact is saved.
            page.locator('#new-run').click()
            page.locator('#prompt').fill('Write a long story about the sea.')
            page.locator('#max-tokens').select_option('128')
            page.locator('#analyze-button').click()
            page.wait_for_function('S.job !== null')
            job_id = page.evaluate('S.job')
            page.locator('#cancel-job').click()
            page.wait_for_function('!S.busy', timeout=60000)
            job = page.request.get(args.url + '/api/jobs/' + job_id).json()
            assert job['state'] == 'cancelled'
            assert 'result' not in job
            passed('live job cancellation')

        page.goto(shared_url)
        page.wait_for_function('S.run && document.getElementById("run-loader").hidden')
        page.locator('[data-lens="jlens"]').click()
        page.locator('#show-special').uncheck()
        page.locator('#toast').wait_for(state='hidden')
        page.screenshot(path=str(args.output / 'viewer-desktop.png'), full_page=True)
        mobile = context.new_page()
        mobile.set_viewport_size({'width': 390, 'height': 844})
        mobile.on('pageerror', lambda e: report['browser_errors'].append(str(e)))
        mobile.goto(shared_url)
        mobile.wait_for_function('S.run && document.getElementById("run-loader").hidden')
        assert mobile.evaluate('document.body.scrollWidth <= innerWidth'), mobile.evaluate('document.body.scrollWidth')
        mobile.locator('#menu-button').click()
        mobile.wait_for_function('document.getElementById("sidebar").getBoundingClientRect().left === 0')
        mobile.locator(f'[data-run="{currency_id}"]').click()
        mobile.wait_for_function('document.getElementById("sidebar").getBoundingClientRect().right <= 0')
        mobile.locator('#intervene-open').click()
        assert mobile.locator('#intervention-dialog').evaluate('(e)=>e.scrollWidth <= e.clientWidth')
        mobile.locator('#intervention-dialog .close-dialog').first.click()
        mobile.screenshot(path=str(args.output / 'viewer-mobile.png'), full_page=True)
        passed('390px phone layout and intervention dialog fit viewport')
        for width in [320, 768, 1024]:
            mobile.set_viewport_size({'width': width, 'height': 900})
            assert mobile.evaluate('document.body.scrollWidth <= innerWidth'), width
        passed('320px, 768px and 1024px responsive widths have no page overflow')
        assert not report['browser_errors'], report['browser_errors']
        passed('no browser JavaScript exceptions')
        browser.close()
    report['completed_at'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    (args.output / 'browser-report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
