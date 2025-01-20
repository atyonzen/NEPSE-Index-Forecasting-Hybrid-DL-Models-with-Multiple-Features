# pip install playwright
# playwright install

import asyncio
import numpy as np
import re
from playwright.async_api import async_playwright


# Function for extracting table data
async def extract_table_data(scraped_data, page):
    # Extract the table data
    rows = await page.query_selector_all('.table-responsive tbody tr')

    # Loop through each row and extract the data
    for row in rows:
        columns = await row.query_selector_all('td')
        row_data = [re.sub(r"[ ,]", "", await cell.inner_text()) for cell in columns]
        scraped_data.append(row_data)
    
    return scraped_data


async def scrape_nepse_indices():
    async with async_playwright() as p:
        args = ["--disable-blink-features=AutomationControlled"]
        # Launch a new browser
        browser = await p.chromium.launch(headless=True, args=args)  # Set headless=False if you want to see the browser
        page = await browser.new_page()

        # Go to the NEPSE index page
        await page.goto('https://nepsealpha.com/nepse-data', wait_until='networkidle')

        # Wait for the necessary content to load
        await page.wait_for_selector('.card-header')  # Wait for the card header to be present

        form_start_date = page.locator('input.form-control.btn-sm')
        await page.evaluate('''document.querySelector('input.form-control.btn-sm').setAttribute('min', '2015-01-01')''')
        await form_start_date.nth(0).fill('2015-01-01')

        filter = page.get_by_role('button', name=' Filter ')
        print(await filter.inner_text())
        await filter.click()
        await page.wait_for_load_state('networkidle')
        await page.wait_for_selector('.table-responsive')  # Wait for the .table-responsive to be present
        page.on('request', lambda data: {
            print(data)
        })

        next_button_visible = True
        page_counter = 1
        scraped_data = []
        if ( page_counter == 1):
            # Extract the table headers
            headers = await page.query_selector_all('.table-responsive thead tr th')
            headers_text = [(await header.inner_text()).replace("'", "") for header in headers]
            scraped_data.append(headers_text)

        # while (next_button_visible):
        #     # increment the page counter
        #     page_counter+=1

        #     # Wait for the table to appear
        #     await page.wait_for_selector('.table-responsive')

        #     # Locate the next button in the pagination
        #     navigation = page.locator('li.pagination-next')

        #     # Check if the next button is disabled
        #     is_disabled = await navigation.get_attribute('class')

        #     if 'disabled' not in is_disabled:
        #         await navigation.click()

        #         # Wait for the page to load completely before moving to the next page
        #         await page.wait_for_load_state('networkidle')

        #         # Extract the table data
        #         await extract_table_data(scraped_data, page)

        #         next_button_visible = True
        #     else:
        #         # No more pages; exit the loop
        #         next_button_visible = False

        # Close the browser
        await browser.close()

        # Save the scraped data to a CSV file
        np.savetxt('nepse_indices.csv', scraped_data, delimiter=',', fmt='%s')
        print("Data saved to nepse_indices.csv")

        # for data in scraped_data:
        #     print(data)

# Run the async function
asyncio.run(scrape_nepse_indices())