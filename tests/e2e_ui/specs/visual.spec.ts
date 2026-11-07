import { test, expect } from '@playwright/test';

test.describe('EvalShield-AI Visual & Responsive Layout Tests', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/index.html');
  });

  test('should display desktop two-column grid layout on standard viewports', async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 800 });

    const main = page.locator('main');
    await expect(main).toBeVisible();

    const evalCard = page.locator('section').first();
    const resultsCard = page.locator('section').nth(1);

    const evalBox = await evalCard.boundingBox();
    const resultsBox = await resultsCard.boundingBox();

    // Verify two columns sit horizontally adjacent
    expect(evalBox).not.toBeNull();
    expect(resultsBox).not.toBeNull();
    if (evalBox && resultsBox) {
      expect(resultsBox.x).toBeGreaterThan(evalBox.x);
    }
  });

  test('should stack cards vertically on mobile viewports', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });

    const evalCard = page.locator('section').first();
    const resultsCard = page.locator('section').nth(1);

    const evalBox = await evalCard.boundingBox();
    const resultsBox = await resultsCard.boundingBox();

    expect(evalBox).not.toBeNull();
    expect(resultsBox).not.toBeNull();
    if (evalBox && resultsBox) {
      expect(resultsBox.y).toBeGreaterThan(evalBox.y);
    }
  });

  test('should render distinct visual alert themes for verdicts', async ({ page }) => {
    // 1. Initial State: Green Allowed
    const banner = page.locator('#verdict-banner');
    await expect(banner).toHaveClass(/banner-allowed/);

    // 2. Toxic State: Red Blocked
    await page.locator('#preset-toxic').click();
    await page.locator('#btn-submit-eval').click();
    await expect(banner).toHaveClass(/banner-blocked/);

    // 3. Drift State: Amber Flagged
    await page.locator('#preset-drift').click();
    await page.locator('#btn-submit-eval').click();
    await expect(banner).toHaveClass(/banner-flagged/);
  });
});
