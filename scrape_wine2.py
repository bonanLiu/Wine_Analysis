import asyncio
from playwright.async_api import async_playwright
import csv

async def scrape():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False) 
        page = await browser.new_page()

        url = "https://www.wineenthusiast.com/?s=port&search_type=ratings"
        await page.goto(url)
        await page.wait_for_timeout(3000)

        
        for _ in range(5):
            await page.mouse.wheel(0, 3000)
            await page.wait_for_timeout(1500)

        blocks = page.locator('div.ratings-block__info')
        count = await blocks.count()
        print(f"Total item: {count}")

        data = []

        for i in range(count):
            block = blocks.nth(i)
            try:
                name = await block.locator('h3.info__title a').text_content() or ""
                link = await block.locator('h3.info__title a').get_attribute("href") or ""
                rating = await block.locator('span.ratings__stars strong').text_content() or ""
                price = await block.locator('span.info__price strong').text_content() or ""
                description = await block.locator('div.info__blurb').text_content() or ""

                data.append({
                    'name': name.strip(),
                    'link': link.strip(),
                    'rating': rating.strip(),
                    'price': price.strip(),
                    'description': description.strip()
                })

            except Exception as e:
                print(f" Error (index {i}): {e}")

        await browser.close()

        if data:
            keys = data[0].keys()
            with open('wineenthusiast_port.csv', 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(data)
            print("Saved CSV!")
        else:
            print("No data!")

asyncio.run(scrape())
