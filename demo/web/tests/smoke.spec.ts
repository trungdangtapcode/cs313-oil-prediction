import { expect, test } from "@playwright/test";

test("renders the mission dashboard from generated ML data", async ({ page }) => {
  await page.goto("/");

  await expect(page.getByRole("heading", { name: "Mission" })).toBeVisible();
  await expect(page.getByText("ENS_FINAL3").first()).toBeVisible();
  await expect(page.getByText("54.8%")).toBeVisible();
  await expect(page.getByText("GCP husky-car")).toBeVisible();
});

test("mobile view exposes the decision microscope tab", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 840 });
  await page.goto("/");
  await page.getByLabel("Microscope").click();

  await expect(page.getByRole("heading", { name: "Microscope" })).toBeVisible();
  await expect(page.getByText("Decision Replay")).toBeVisible();
});
