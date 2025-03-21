import joblib
import os

# Get the absolute path of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Use the correct relative path to the model directory
model_path = os.path.join(current_dir, "../model/model.pkl")
vectorizer_path = os.path.join(current_dir, "../model/vectorizer.pkl")

# Load trained model and vectorizer
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

def predict_news(news_text):
    '''Predict whether a given news article is Fake or Real.'''
    text_vectorised = vectorizer.transform([news_text])
    prediction = model.predict(text_vectorised)[0]
    return "Fake News" if prediction == 0 else "Real News"

if __name__ == "__main__":
    news_text = input("LAS VEGAS - As Republican presidential nominee Donald Trump prepared to go head-to-head with Democratic rival Hillary Clinton in their third and final debate in Las Vegas on Wednesday, hundreds of hospitality workers and protesters gathered outside the hotel Trump co-owns near the Las Vegas Strip.  Demonstrators waved signs and banners next to what organizers called a â€œwallâ€ of taco trucks. That was a reference to Trumpâ€™s proposal to build a wall on the U.S. border with Mexico, as well as a riff on comments by Trump supporter Marco Gutierrez who said that without action on immigration reform, â€œYouâ€™re going to have taco trucks on every corner.â€ â€œWe have fought for our rights and we donâ€™t want Donald Trump to take them away from us,â€ said Las Vegas resident Miguel Faria. â€œIf this man gets the presidency then everything will be ruined.â€  Several protesters wore sashes printed with the phrases â€œMiss Piggyâ€ and â€œMiss Housekeeping,â€ using the names that Clinton has said Trump called a former beauty queen who had gained weight and who was Latina. The protest was organized by the Culinary Union, which represents about 57,000 workers in Nevada, the majority of whom are Latino. Luis Hernandez, a musician with the norteÃ±o band â€œLos Tigres del Norte,â€ encouraged Latino voters to head to the polls on Nov. 8.  â€œWe canâ€™t just go on hoping someone will vote on our behalf,â€ he told Reuters. â€œWe need to go out and vote because the Hispanic vote is going to make the difference in these elections.â€ Among the speakers at the protest was civil rights leader the Rev. Jesse Jackson, who said he hoped the candidates would stick to policies at Wednesdayâ€™s debate, instead of attacking each other. â€œWe all as Americans live under one big tent. Hillary represents that higher ground. I hope that tonight they will not wallow in snake politics but will fly like eagles and take us all to higher ground,â€ Jackson told Reuters.  According to Bethany Khan, the Culinary Unionâ€™s communications director, workers at the Trump International Hotel voted to unionize in December 2015 but still do not have a contract.  Some protesters blamed that on Trump, who owns 50 percent of the property.  â€œHe says heâ€™s the greatest negotiator but heâ€™s not coming to the table to support the workers that give him money and make a profit for him,â€ said Maria Teresa Liedermann.")
    print("\nPrediction:", predict_news(news_text))

