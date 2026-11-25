import { test, expect } from '@playwright/test';

test.describe('EvalShield-AI Dashboard E2E Workflow', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/index.html?mode=simulation');
  });

  test('should load page with correct branding and initial operational state', async ({ page }) => {
    await expect(page).toHaveTitle(/EvalShield-AI/);
    const brand = page.locator('.brand-title');
    await expect(brand).toContainText('EvalShield-AI');

    const statusDot = page.locator('#system-status-dot');
    await expect(statusDot).toBeVisible();

    const decisionBanner = page.locator('#decision-text');
    await expect(decisionBanner).toContainText('ALLOWED');
  });

  test('should synchronize slider and number input fields accurately', async ({ page }) => {
    const slider = page.locator('#toxicity-slider');
    const numInput = page.locator('#toxicity-num');
    const display = page.locator('#toxicity-val-display');

    // Fill number input directly
    await numInput.fill('0.45');
    await numInput.dispatchEvent('input');

    await expect(slider).toHaveValue('0.45');
    await expect(display).toHaveText('0.45');
  });

  test('should trigger prompt injection preset and receive BLOCKED verdict', async ({ page }) => {
    // Click Prompt Injection preset button
    const injectionPresetBtn = page.locator('#preset-injection');
    await injectionPresetBtn.click();

    // Verify fields populated
    const injectionSlider = page.locator('#injection-slider');
    await expect(injectionSlider).toHaveValue('0.92');

    // Submit evaluation
    const submitBtn = page.locator('#btn-submit-eval');
    await submitBtn.click();

    // Check decision banner
    const verdict = page.locator('#decision-text');
    await expect(verdict).toHaveText('BLOCKED', { timeout: 5000 });

    const verdictBanner = page.locator('#verdict-banner');
    await expect(verdictBanner).toHaveClass(/banner-blocked/);

    // Violations should be listed
    const violationsList = page.locator('#violations-list');
    await expect(violationsList).toBeVisible();
    await expect(violationsList).toContainText('injection');
  });

  test('should trigger benign preset and clear violation notices', async ({ page }) => {
    // Select benign preset
    await page.locator('#preset-benign').click();
    await page.locator('#btn-submit-eval').click();

    const verdict = page.locator('#decision-text');
    await expect(verdict).toHaveText('ALLOWED');

    const noViolationsMsg = page.locator('#no-violations-msg');
    await expect(noViolationsMsg).toBeVisible();
    await expect(page.locator('#violations-list')).toBeHidden();
  });

  test('explicit simulation mode is persistent and labels generated results', async ({ page }) => {
    let apiRequests = 0;
    await page.route('**/evaluate', (route) => {
      apiRequests += 1;
      return route.abort();
    });
    await page.route('**/red-team/simulate', (route) => {
      apiRequests += 1;
      return route.abort();
    });
    await expect(page.locator('#simulation-mode-notice')).toBeVisible();
    await page.locator('#preset-injection').click();
    await page.locator('#btn-submit-eval').click();

    await expect(page.locator('#engine-display')).toHaveText('Local simulation (client-side)');

    await page.locator('#btn-launch-redteam').click();
    await expect(page.locator('#redteam-results-box')).toBeVisible();
    await expect(page.locator('#redteam-asr-badge')).toContainText('simulation only');
    await expect(page.locator('#redteam-receipt')).toContainText('ES-SIG-CLIENT-FALLBACK');
    await expect(page.locator('#simulation-mode-notice')).toBeVisible();
    expect(apiRequests).toBe(0);
  });
});

test.describe('Server-backed dashboard API contract', () => {
  const evaluationResponse = {
    shield_decision: 'BLOCKED',
    risk_score: 0.9,
    confidence: 0.91,
    backend: 'test-api',
    latency_ms: 12.3,
    violation_reasons: ['Prompt injection risk'],
    canary_leak_detected: false,
    leaked_canary_token: null,
    pii_detected: [],
    risk_contributions: {
      prompt_injection_score: 0.5,
      toxicity_score: 0.2,
      hallucination_risk: 0.1,
      semantic_drift: 0.1,
      token_entropy: 0.1,
    },
  };

  test('evaluation uses the same-origin /evaluate route', async ({ page }) => {
    await page.route('**/evaluate', (route) =>
      route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(evaluationResponse) })
    );
    await page.goto('/index.html');

    const requestPromise = page.waitForRequest((request) => request.url().endsWith('/evaluate'));
    await page.locator('#btn-submit-eval').click();
    const request = await requestPromise;

    expect(new URL(request.url()).origin).toBe(new URL(page.url()).origin);
    expect(new URL(request.url()).pathname).toBe('/evaluate');
    expect(request.headers().authorization).toBeUndefined();
    await expect(page.locator('#decision-text')).toHaveText('BLOCKED');
    await expect(page.locator('#engine-display')).toHaveText('test-api');
    await expect(page.locator('#simulation-mode-notice')).toBeHidden();
    const metricsHref = await page.locator('a[data-api-path="/metrics"]').evaluate((link) => link.href);
    expect(metricsHref).toBe(`${new URL(page.url()).origin}/metrics`);
  });

  test('red-team action uses the same-origin /red-team/simulate route', async ({ page }) => {
    const response = {
      campaign_id: 'test-campaign',
      seed_prompt: 'test seed',
      total_mutations: 1,
      evasion_count: 0,
      attack_success_rate: 0,
      attack_vectors: [{
        vector_id: 'v1',
        mutation_type: 'roleplay_frame',
        original_prompt: 'test seed',
        mutated_prompt: 'mutated seed',
        shield_decision: 'BLOCKED',
        risk_score: 0.9,
        evasion_succeeded: false,
        violation_reasons: [],
        latency_ms: 1,
      }],
      discovered_bypasses: [],
      cryptographic_receipt: 'test-signature',
      recalibration_triggered: false,
    };
    await page.route('**/red-team/simulate', (route) =>
      route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(response) })
    );
    await page.goto('/index.html');

    const requestPromise = page.waitForRequest((request) => request.url().endsWith('/red-team/simulate'));
    await page.locator('#btn-launch-redteam').click();
    const request = await requestPromise;

    expect(new URL(request.url()).origin).toBe(new URL(page.url()).origin);
    expect(new URL(request.url()).pathname).toBe('/red-team/simulate');
    expect(request.headers().authorization).toBeUndefined();
    await expect(page.locator('#redteam-results-box')).toBeVisible();
    await expect(page.locator('#redteam-receipt')).toContainText('test-signature');
  });

  for (const status of [401, 403]) {
    test(`evaluation HTTP ${status} is shown and does not produce a result`, async ({ page }) => {
      await page.route('**/evaluate', (route) =>
        route.fulfill({ status, contentType: 'application/json', body: JSON.stringify({ detail: 'denied' }) })
      );
      await page.goto('/index.html');
      await page.locator('#btn-submit-eval').click();

      await expect(page.locator('#evaluation-api-error')).toBeVisible();
      await expect(page.locator('#evaluation-api-error')).toContainText(status === 401 ? 'authentication' : 'authorization');
      await expect(page.locator('#evaluation-result-content')).toBeHidden();
    });
  }

  test('server and network failures are shown without invoking simulation fallback', async ({ page }) => {
    await page.route('**/evaluate', (route) =>
      route.fulfill({ status: 500, contentType: 'application/json', body: JSON.stringify({ detail: 'failed' }) })
    );
    await page.goto('/index.html');
    await page.locator('#btn-submit-eval').click();

    await expect(page.locator('#evaluation-api-error')).toContainText('HTTP 500');
    await expect(page.locator('#evaluation-result-content')).toBeHidden();

    await page.unroute('**/evaluate');
    await page.route('**/evaluate', (route) => route.abort('failed'));
    await page.locator('#btn-submit-eval').click();

    await expect(page.locator('#evaluation-api-error')).toContainText('Could not reach the API');
    await expect(page.locator('#evaluation-result-content')).toBeHidden();
    await expect(page.locator('#simulation-mode-notice')).toBeHidden();
  });

  test('red-team API failure does not fabricate a defensive campaign', async ({ page }) => {
    await page.route('**/red-team/simulate', (route) =>
      route.fulfill({ status: 401, contentType: 'application/json', body: JSON.stringify({ detail: 'denied' }) })
    );
    await page.goto('/index.html');
    await page.locator('#btn-launch-redteam').click();

    await expect(page.locator('#redteam-api-error')).toContainText('authentication');
    await expect(page.locator('#redteam-results-box')).toBeHidden();
    await expect(page.locator('#redteam-asr-badge')).not.toContainText('100% Defended');
    await expect(page.locator('#simulation-mode-notice')).toBeHidden();
  });
});
