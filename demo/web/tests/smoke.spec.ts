import { expect, test } from "@playwright/test";

test("renders the mission dashboard from generated ML data", async ({ page }) => {
  await page.goto("/");

  await expect(page.getByRole("heading", { name: "Mission" })).toBeVisible();
  await expect(page.getByText("ENS_FINAL3").first()).toBeVisible();
  await expect(page.getByText("54.8%")).toBeVisible();
  await expect(page.getByText("Cloud Run live")).toBeVisible();
  await expect(page.getByText("Android APK")).toBeVisible();
  await expect(page.getByRole("link", { name: "Download APK" })).toHaveAttribute(
    "href",
    /\/downloads\/oil-signal-mine-latest\.apk$/,
  );
});

test("mobile view exposes the decision microscope tab", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 840 });
  await page.goto("/");
  await page.getByLabel("Microscope").click();

  await expect(page.getByRole("heading", { name: "Microscope" })).toBeVisible();
  await expect(page.getByText("Decision Replay")).toBeVisible();
});

test("renders the trading research page from generated strategy data", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Trading" }).click();

  await expect(page.getByRole("heading", { name: "Trading", exact: true })).toBeVisible();
  await expect(page.getByText("Equity Curve")).toBeVisible();
  await expect(page.getByText("Threshold Profit Lab")).toBeVisible();
  await expect(page.getByText("Profit on")).toBeVisible();
  await expect(page.getByText("Buy & Hold").first()).toBeVisible();
  await expect(page.getByText("Transaction cost")).toBeVisible();
});
