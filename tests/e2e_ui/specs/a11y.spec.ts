import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';

test.describe('EvalShield-AI Accessibility (WCAG 2.2 AA Compliance)', () => {
  test('should pass automated axe-core audit with zero violations on initial load', async ({ page }) => {
    await page.goto('/index.html');

    const accessibilityScanResults = await new AxeBuilder({ page })
      .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa'])
      .analyze();

    expect(accessibilityScanResults.violations).toEqual([]);
  });

  test('should remain accessible after dynamic state transition to BLOCKED verdict', async ({ page }) => {
    await page.goto('/index.html');

    // Trigger state transition
    await page.locator('#preset-injection').click();
    await page.locator('#btn-submit-eval').click();

    // Wait for the BLOCKED status to render
    await expect(page.locator('#decision-text')).toHaveText('BLOCKED');

    const accessibilityScanResults = await new AxeBuilder({ page })
      .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa'])
      .analyze();

    expect(accessibilityScanResults.violations).toEqual([]);
  });

  test('should support keyboard navigation across interactive controls', async ({ page }) => {
    await page.goto('/index.html');

    // Start tabbing from beginning
    await page.keyboard.press('Tab');
    const firstFocusedTag = await page.evaluate(() => document.activeElement?.tagName);
    expect(['BUTTON', 'A', 'INPUT']).toContain(firstFocusedTag);

    // Tab into submit button and press Enter
    const submitBtn = page.locator('#btn-submit-eval');
    await submitBtn.focus();
    await page.keyboard.press('Enter');

    // Verify form responded to keyboard interaction
    await expect(page.locator('#decision-text')).toBeVisible();
  });
});
