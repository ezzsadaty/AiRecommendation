import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

places_df = pd.read_csv('places_data.csv')
restaurants_df = pd.read_csv('restaurants_data.csv')
hotels_df = pd.read_csv('Hotels_data.csv')

# Label Encoding for categorical data
le_place_type = LabelEncoder()
le_place_region = LabelEncoder()
le_place_budget = LabelEncoder()
le_place_activity = LabelEncoder()
le_place_indoor_outdoor = LabelEncoder()

le_rest_food = LabelEncoder()
le_rest_region = LabelEncoder()
le_rest_price_range = LabelEncoder()
le_rest_ambiance = LabelEncoder()

le_hotel_type = LabelEncoder()
le_hotel_region = LabelEncoder()
le_hotel_budget = LabelEncoder()
le_hotel_facilities = LabelEncoder()
le_hotel_popularity = LabelEncoder()

places_df['Type_encoded'] = le_place_type.fit_transform(places_df['Type'])
places_df['Region_encoded'] = le_place_region.fit_transform(places_df['Region'])
places_df['Budget_encoded'] = le_place_budget.fit_transform(places_df['Budget'])
places_df['Activity_Level_encoded'] = le_place_activity.fit_transform(places_df['Activity_Level'])
places_df['Indoor_Outdoor_encoded'] = le_place_indoor_outdoor.fit_transform(places_df['Indoor_Outdoor'])

restaurants_df['Food_encoded'] = le_rest_food.fit_transform(restaurants_df['Food'])
restaurants_df['Region_encoded'] = le_rest_region.fit_transform(restaurants_df['Region'])
restaurants_df['Price_Range_encoded'] = le_rest_price_range.fit_transform(restaurants_df['Price_Range'])
restaurants_df['Ambiance_encoded'] = le_rest_ambiance.fit_transform(restaurants_df['Ambiance'])

hotels_df['Type_encoded'] = le_hotel_type.fit_transform(hotels_df['Type'])
hotels_df['Region_encoded'] = le_hotel_region.fit_transform(hotels_df['Region'])
hotels_df['Budget_encoded'] = le_hotel_budget.fit_transform(hotels_df['Budget'])
hotels_df['Facilities_encoded'] = le_hotel_facilities.fit_transform(hotels_df['Facilities'])
hotels_df['Popularity_encoded'] = le_hotel_popularity.fit_transform(hotels_df['Popularity'])

# Train models for places, restaurants, and hotels
X_places = places_df[['ID','Type_encoded', 'Region_encoded', 'Rating', 'Popularity', 'Budget_encoded', 'Activity_Level_encoded', 'Indoor_Outdoor_encoded']]
y_places = places_df['Place']
X_train_places, X_test_places, y_train_places, y_test_places = train_test_split(X_places, y_places, test_size=0.2, random_state=42)
place_model = RandomForestClassifier(random_state=42)
place_model.fit(X_train_places.drop(columns=['ID']), y_train_places)

X_restaurants = restaurants_df[['ID','Food_encoded', 'Region_encoded', 'Rating', 'Popularity', 'Price_Range_encoded', 'Ambiance_encoded']]
y_restaurants = restaurants_df['Place']
X_train_restaurants, X_test_restaurants, y_train_restaurants, y_test_restaurants = train_test_split(X_restaurants, y_restaurants, test_size=0.2, random_state=42)
restaurant_model = RandomForestClassifier(random_state=42)
restaurant_model.fit(X_train_restaurants.drop(columns=['ID']), y_train_restaurants)

X_hotels = hotels_df[['ID','Type_encoded', 'Region_encoded', 'Rating', 'Popularity_encoded', 'Budget_encoded', 'Facilities_encoded']]
y_hotels = hotels_df['Hotels']
X_train_hotels, X_test_hotels, y_train_hotels, y_test_hotels = train_test_split(X_hotels, y_hotels, test_size=0.2, random_state=42)
hotel_model = RandomForestClassifier(random_state=42)
hotel_model.fit(X_train_hotels.drop(columns=['ID']), y_train_hotels)

class RequestHandler(BaseHTTPRequestHandler):
    def _send_response(self, response_data, status=200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(json.dumps(response_data).encode('utf-8'))

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_POST(self):
        if self.path == '/get_recommendations':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            request_data = json.loads(post_data)

            # Extract user input from request
            region = request_data.get('region')
            type_place = request_data.get('placeType')
            type_food = request_data.get('foodType')

            # Encode user inputs
            type_place_encoded = le_place_type.transform([type_place])[0]
            type_food_encoded = le_rest_food.transform([type_food])[0]
            region_encoded = le_place_region.transform([region])[0]

            # Number of recommendations to make
            n_recommendations = 5

            # Prepare response dictionary
            response = {
                "places": [],
                "restaurants": [],
                "hotels": []
            }

            # Filter places to only include the specified region
            filtered_places_df = places_df[places_df['Region_encoded'] == region_encoded]
            if not filtered_places_df.empty:
                input_place_df = pd.DataFrame([[0, type_place_encoded, region_encoded, filtered_places_df['Rating'].mean(), filtered_places_df['Popularity'].mean(), filtered_places_df['Budget_encoded'].mean(), filtered_places_df['Activity_Level_encoded'].mean(), filtered_places_df['Indoor_Outdoor_encoded'].mean()]], columns=['ID', 'Type_encoded', 'Region_encoded', 'Rating', 'Popularity', 'Budget_encoded', 'Activity_Level_encoded', 'Indoor_Outdoor_encoded'])
                probas = place_model.predict_proba(input_place_df.drop(columns=['ID']))
                top_indices = np.argsort(probas[0])[-n_recommendations:][::-1]
                top_suggestions = place_model.classes_[top_indices]

                for suggestion in top_suggestions:
                    suggested_place_details = places_df[places_df['Place'] == suggestion].drop(columns=['Type_encoded', 'Region_encoded', 'Budget_encoded', 'Activity_Level_encoded', 'Indoor_Outdoor_encoded'])
                    if suggested_place_details['Region'].iloc[0] == region and suggested_place_details['Type'].iloc[0] == type_place:
                        response["places"].append(suggested_place_details.to_dict('records')[0])

            # Filter restaurants to only include the specified region
            filtered_restaurants_df = restaurants_df[restaurants_df['Region_encoded'] == region_encoded]
            if not filtered_restaurants_df.empty:
                input_restaurant_df = pd.DataFrame([[0, type_food_encoded, region_encoded, filtered_restaurants_df['Rating'].mean(), filtered_restaurants_df['Popularity'].mean(), filtered_restaurants_df['Price_Range_encoded'].mean(), filtered_restaurants_df['Ambiance_encoded'].mean()]], columns=['ID', 'Food_encoded', 'Region_encoded', 'Rating', 'Popularity', 'Price_Range_encoded', 'Ambiance_encoded'])
                probas = restaurant_model.predict_proba(input_restaurant_df.drop(columns=['ID']))
                top_indices = np.argsort(probas[0])[-n_recommendations:][::-1]
                top_recommendations = restaurant_model.classes_[top_indices]

                for recommendation in top_recommendations:
                    recommended_restaurant_details = restaurants_df[restaurants_df['Place'] == recommendation].drop(columns=['Food_encoded', 'Region_encoded', 'Price_Range_encoded', 'Ambiance_encoded'])
                    if recommended_restaurant_details['Region'].iloc[0] == region and recommended_restaurant_details['Food'].iloc[0] == type_food:
                        response["restaurants"].append(recommended_restaurant_details.to_dict('records')[0])

            # Predict hotels based on region
            filtered_hotels_df = hotels_df[hotels_df['Region_encoded'] == region_encoded]
            if not filtered_hotels_df.empty:
                input_hotel_df = pd.DataFrame([[0, filtered_hotels_df['Type_encoded'].mean(), region_encoded, filtered_hotels_df['Rating'].mean(), filtered_hotels_df['Popularity_encoded'].mean(), filtered_hotels_df['Budget_encoded'].mean(), filtered_hotels_df['Facilities_encoded'].mean()]], columns=['ID', 'Type_encoded', 'Region_encoded', 'Rating', 'Popularity_encoded', 'Budget_encoded', 'Facilities_encoded'])
                probas = hotel_model.predict_proba(input_hotel_df.drop(columns=['ID']))
                top_indices = np.argsort(probas[0])[-n_recommendations:][::-1]
                top_recommendations = hotel_model.classes_[top_indices]

                for recommendation in top_recommendations:
                    recommended_hotel_details = hotels_df[hotels_df['Hotels'] == recommendation].drop(columns=['Type_encoded', 'Region_encoded', 'Budget_encoded', 'Facilities_encoded', 'Popularity_encoded'])
                    if recommended_hotel_details['Region'].iloc[0] == region:
                        response["hotels"].append(recommended_hotel_details.to_dict('records')[0])

            self._send_response(response)
        else:
            self._send_response({'error': 'Invalid endpoint'}, status=404)

def run(server_class=HTTPServer, handler_class=RequestHandler, port=8080):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f'Starting httpd server on port {port}')
    httpd.serve_forever()

if __name__ == '__main__':
    run()
