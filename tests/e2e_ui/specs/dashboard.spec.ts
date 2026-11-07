import { test, expect } from '@playwright/test';

test.describe('EvalShield-AI Dashboard E2E Workflow', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/index.html');
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
});
