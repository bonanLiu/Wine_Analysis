from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import pandas as pd
import time

def scrape_wine_searcher():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)  # Set False to see the browser in action
        context = browser.new_context()
        page = context.new_page()

        url = "https://www.wine-searcher.com/regions-portugal?tab_F=mostpopular#winesortlist"
        page.goto(url)

        # Wait for page to load
        time.sleep(10)

        # Extract page content
        html = page.content()
        browser.close()

    # Parse HTML with BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")

    # Store extracted wine data
    wine_data = []

    # Find all wine containers
    wine_items = soup.find_all('td', class_='col-9 col-sm-10 col-md-11 d-block d-md-flex flex-column flex-md-row pl-1 pr-0')

    print(f"Found {len(wine_items)} wine items!")

    # Loop through each wine entry
    for item in wine_items:
        # Extract wine name
        name_tag = item.find('a', class_='superlative-list__name font-light-bold')
        name = name_tag.get_text(strip=True) 

        # Extract wine type
        type_tag = item.find('a', href=lambda x: x and 'grape' in x)
        wine_type = type_tag.get_text(strip=True) 

        # Extract popularity
        pop_div = item.find('div', class_='text-md-center superlative-list-md-w-16 order-md-2 mb-1 mb-md-0')
        popularity = pop_div.get_text(strip=True).replace("in popularity", "").strip() 

        # Extract price
        price_div = item.find('div', class_='text-md-center superlative-list-md-w-15 order-md-4 mb-1 mb-md-0')
        price_value = price_div.find('b', class_='font-light-bold') 
        price = f"CA$ {price_value.get_text(strip=True)}" 

        # Extract rating
        rating_div = item.find('div', class_='text-md-center superlative-list-md-w-16 order-md-3 superlative-list__score')
        rating_tag = rating_div.find('span', class_='badge badge-pill badge-rating') 
        rating = rating_tag.get_text(strip=True) 

        # Append data to the list
        wine_data.append({
            'Name': name,
            'Type': wine_type,
            'Popularity': popularity,
            'Price': price,
            'Rating': rating
        })

    # Convert to DataFrame
    df = pd.DataFrame(wine_data)

    # Show DataFrame
    print(df)

    # Save to CSV file
    df.to_csv('wine_searcher_data_mostpopular.csv', index=False)

# Run the scraper
scrape_wine_searcher()
