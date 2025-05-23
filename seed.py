from backend.database import db
from backend.models import Category, Word
from app import create_app

WORDS = {
    "Nourriture": [
        "ship",
        "bird",
        "camel",
        "cat",
        "dolphin",
        "crab",
        "fish",
        "flamingo",
        "hedgehog",
        "raccoon",
        "lion",
        "octopus",
        "whale",
        "shark",
        "rhinoceros",
        "rabbit",
        "pig",
        "crocodile",
        "cow",
        "elephant"
    ],

    "Objet": [
        "alarm_clock",
        "anvil",
        "axe",
        "backpack",
        "baseball_bat",
        "bed",
        "belt",
        "bicycle",
        "cell_phone",
        "flip_flops",
        "headphones",
        "tshirt",
        "flower",
        "eyeglasses",
        "harp",
        "hexagon",
        "key",
        "ladder",
        "knife",
        "fork"
    ],
    "Animaux": [
        "apple",
        "banana",
        "birthday_cake",
        "blueberry",
        "bread",
        "broccoli",
        "carrot",
        "cookie",
        "donut",
        "grapes",
        "hamburger",
        "hot_dog",
        "ice_cream",
        "lollipop",
        "mushroom",
        "pear",
        "pineapple",
        "pizza",
        "strawberry",
        "watermelon"
    ],
}


def run():
    app = create_app()
    with app.app_context():
        for cat_name, words in WORDS.items():
            cat = Category(name=cat_name)
            db.session.add(cat)
            db.session.flush()
            for w in words:
                db.session.add(Word(text=w, category_id=cat.id))
        db.session.commit()
        print("Seed OK")


if __name__ == "__main__":
    run()
