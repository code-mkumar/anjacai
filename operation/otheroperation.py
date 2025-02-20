from datetime import datetime
import random

def get_dynamic_greeting():
    hour = datetime.now().hour
    if hour < 12:
        return "Good Morning! Welcome to Anjac AI."
    elif 12 <= hour < 16:
        return "Good Afternoon! Glad to see you here."
    else:
        return "Good Evening! How can I assist you today?"

# Function to display a random fun fact
def get_fun_fact():
    fun_facts = [
        
        "🏫 History in a Name: ANJAC is named after the generous founders Thiru P. Ayya Nadar and Thirumathi A. Janaki Ammal",
        "🌏 Little Japan Connection: Located in Sivakasi, also known as Little Japan, ANJAC has a legacy of excellence since 1963",
        "🌳 Vast and Vibrant Campus: Spread over 157 acres, ANJAC is a self-sufficient educational hub",
        "🏆 NIRF Top 100: Ranked 69th in NIRF 2023, showcasing academic and research excellence",
        "✨ NAAC ‘A+’ Excellence: Re-accredited with a stellar CGPA of 3.48 by NAAC",
        "🍄 Mushroom Centre Marvel: ANJAC features its own Mushroom Cultivation Centre",
        "♻️ Eco-Friendly Practices: The campus recycles lab water through a Water Treatment Grid",
        "📶 Wi-Fi Wonderland: Entire campus enjoys blazing-fast Wi-Fi at 100 Mbps",
        "🚀 Student Startup Hub: ANJAC’s All-Hub encourages innovative ideas and entrepreneurship",
        "📚 Digital Library: Access over 1 lakh books and 110 journals through its advanced digital library",
        "🦯 Inclusive Tech: Braille materials and assistive technologies support visually challenged students",
        "🏅 Sports Powerhouse: A 50-bed UGC Sports Hostel nurtures athletic talent",
        "📻 Community Radio: ANJA Community Radio connects and educates the surrounding community",
        "🌈 Dynamic Diversity: With 36 academic associations, there’s something for everyone",
        "🌿 Nature at Heart: Tree-growing competitions and an ornamental garden promote eco-awareness",
        "📜 Certificate Extravaganza: Offers 39 certificate courses ranging from Animation to Tourism",
        "🚨 Anti-Ragging Pledge: ANJAC has a 24x7 anti-ragging helpline for a safe campus",
        "🌱 Green Goals: Initiatives like vermicomposting and mushroom cultivation lead sustainability efforts",
        "🎭 Cultural Competitions: From Fresher’s Day to national events, ANJAC celebrates talent in style",
        "💻 Tech Titans: Boasts over 600 high-configured computers and cutting-edge software for futuristic learning",
    ]
    return random.choice(fun_facts)