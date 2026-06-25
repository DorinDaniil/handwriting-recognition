"""Punctuated reference texts for all of CVL's samples (public domain), verbatim:
six English (Flatland, Poe, Wilde, Darwin, Mailüfterl, Macbeth) + German Faust.

CVL transcribes words only (no punctuation). Recovery matches each line to the best-fitting
reference (by word coverage) and copies that reference's punctuation onto the line's own
words — no marks are invented; an unmatched line stays words-only.
"""

REFERENCES = [
    # Edwin A. Abbott — Flatland
    "Imagine a vast sheet of paper on which straight Lines, Triangles, Squares, "
    "Pentagons, Hexagons, and other figures, instead of remaining fixed in their places, "
    "move freely about, on or in the surface, but without the power of rising above or "
    "sinking below it, very much like shadows—only hard and with luminous edges—and you "
    "will then have a pretty correct notion of my country and countrymen. Alas, a few "
    "years ago, I should have said \"my universe\": but now my mind has been opened to "
    "higher views of things.",

    # Edgar Allan Poe — The Fall of the House of Usher
    "While I gazed, this fissure rapidly widened—there came a fierce breath of the "
    "whirlwind—the entire orb of the satellite burst at once upon my sight—my brain "
    "reeled as I saw the mighty walls rushing asunder—there was a long tumultuous "
    "shouting sound like the voice of a thousand waters—and the deep and dank tarn at my "
    "feet closed sullenly and silently over the fragments of the \"House of Usher\".",

    # Oscar Wilde — The Picture of Dorian Gray (Chapter 8, Dorian to Sibyl Vane)
    "You have killed my love. You used to stir my imagination. Now you don't even stir my "
    "curiosity. You simply produce no effect. I loved you because you were marvellous, "
    "because you had genius and intellect, because you realised the dreams of great poets "
    "and gave shape and substance to the shadows of art. You have thrown it all away. You "
    "are shallow and stupid.",

    # Charles Darwin — On the Origin of Species (Chapter 1, opening)
    "When we look to the individuals of the same variety or sub-variety of our older "
    "cultivated plants and animals, one of the first points which strikes us is, that they "
    "generally differ much more from each other than do the individuals of any one species "
    "or variety in a state of nature.",

    # Wikipedia — Mailüfterl (English article)
    "Mailüfterl is an Austrian nickname for the first computer working solely on "
    "transistors on the European mainland. It was built in 1955 at the Vienna University of "
    "Technology by Heinz Zemanek. The builder plays on a quote, on an operating computer, "
    "if it is not the speed that Whirlwind or Typhoon can achieve, it will at least be a "
    "gentle Viennese Mailüfterl.",

    # William Shakespeare — Macbeth (Act 1, Scene 2)
    "The merciless Macdonwald—worthy to be a rebel, for to that the multiplying villanies "
    "of nature do swarm upon him—from the western isles of kerns and gallowglasses is "
    "supplied; and fortune, on his damned quarrel smiling, show'd like a rebel's whore: "
    "but all's too weak; for brave Macbeth—well he deserves that name—disdaining fortune, "
    "with his brandish'd steel, which smoked with bloody execution, like valour's minion "
    "carved out his passage till he faced the slave; which ne'er shook hands, nor bade "
    "farewell to him, till he unseam'd him from the nave to the chops, and fix'd his head "
    "upon our battlements.",

    # Johann Wolfgang von Goethe — Faust (the wager)
    "Werd ich zum Augenblicke sagen: Verweile doch! du bist so schön! Dann magst du mich "
    "in Fesseln schlagen, Dann will ich gern zu Grunde gehn!",
]
