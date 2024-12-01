import datasets

dataset = datasets.load_dataset("hotpot_qa", "distractor")

print(dataset["train"][10])

# https://huggingface.co/datasets/hotpotqa/hotpot_qa?row=16
# Paper: https://arxiv.org/pdf/1809.09600
row1 = {
    "id": "5a7a06935542990198eaf050",
    "question": "Which magazine was started first Arthur's Magazine or First for Women?",
    "answer": "Arthur's Magazine",
    "type": "comparison",
    "level": "medium",
    "supporting_facts": {
        "title": ["Arthur's Magazine", "First for Women"],
        "sent_id": [0, 0],
    },
    "context": {
        "title": [
            "Radio City (Indian radio station)",
            "History of Albanian football",
            "Echosmith",
            "Women's colleges in the Southern United States",
            "First Arthur County Courthouse and Jail",
            "Arthur's Magazine",
            "2014–15 Ukrainian Hockey Championship",
            "First for Women",
            "Freeway Complex Fire",
            "William Rast",
        ],
        "sentences": [
            [
                "Radio City is India's first private FM radio station and was started on 3 July 2001.",
                " It broadcasts on 91.1 (earlier 91.0 in most cities) megahertz from Mumbai (where it was started in 2004), Bengaluru (started first in 2001), Lucknow and New Delhi (since 2003).",
                " It plays Hindi, English and regional songs.",
                " It was launched in Hyderabad in March 2006, in Chennai on 7 July 2006 and in Visakhapatnam October 2007.",
                " Radio City recently forayed into New Media in May 2008 with the launch of a music portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features.",
                " The Radio station currently plays a mix of Hindi and Regional music.",
                " Abraham Thomas is the CEO of the company.",
            ],
            [
                "Football in Albania existed before the Albanian Football Federation (FSHF) was created.",
                " This was evidenced by the team's registration at the Balkan Cup tournament during 1929-1931, which started in 1929 (although Albania eventually had pressure from the teams because of competition, competition started first and was strong enough in the duels) .",
                " Albanian National Team was founded on June 6, 1930, but Albania had to wait 16 years to play its first international match and then defeated Yugoslavia in 1946.",
                " In 1932, Albania joined FIFA (during the 12–16 June convention ) And in 1954 she was one of the founding members of UEFA.",
            ],
            [
                "Echosmith is an American, Corporate indie pop band formed in February 2009 in Chino, California.",
                " Originally formed as a quartet of siblings, the band currently consists of Sydney, Noah and Graham Sierota, following the departure of eldest sibling Jamie in late 2016.",
                ' Echosmith started first as "Ready Set Go!"',
                " until they signed to Warner Bros.",
                " Records in May 2012.",
                ' They are best known for their hit song "Cool Kids", which reached number 13 on the "Billboard" Hot 100 and was certified double platinum by the RIAA with over 1,200,000 sales in the United States and also double platinum by ARIA in Australia.',
                " The song was Warner Bros.",
                " Records' fifth-biggest-selling-digital song of 2014, with 1.3 million downloads sold.",
                ' The band\'s debut album, "Talking Dreams", was released on October 8, 2013.',
            ],
            [
                "Women's colleges in the Southern United States refers to undergraduate, bachelor's degree–granting institutions, often liberal arts colleges, whose student populations consist exclusively or almost exclusively of women, located in the Southern United States.",
                " Many started first as girls' seminaries or academies.",
                " Salem College is the oldest female educational institution in the South and Wesleyan College is the first that was established specifically as a college for women.",
                " Some schools, such as Mary Baldwin University and Salem College, offer coeducational courses at the graduate level.",
            ],
            [
                "The First Arthur County Courthouse and Jail, was perhaps the smallest court house in the United States, and serves now as a museum."
            ],
            [
                "Arthur's Magazine (1844–1846) was an American literary periodical published in Philadelphia in the 19th century.",
                " Edited by T.S. Arthur, it featured work by Edgar A. Poe, J.H. Ingraham, Sarah Josepha Hale, Thomas G. Spear, and others.",
                " In May 1846 it was merged into \"Godey's Lady's Book\".",
            ],
            [
                "The 2014–15 Ukrainian Hockey Championship was the 23rd season of the Ukrainian Hockey Championship.",
                " Only four teams participated in the league this season, because of the instability in Ukraine and that most of the clubs had economical issues.",
                " Generals Kiev was the only team that participated in the league the previous season, and the season started first after the year-end of 2014.",
                " The regular season included just 12 rounds, where all the teams went to the semifinals.",
                " In the final, ATEK Kiev defeated the regular season winner HK Kremenchuk.",
            ],
            [
                "First for Women is a woman's magazine published by Bauer Media Group in the USA.",
                " The magazine was started in 1989.",
                " It is based in Englewood Cliffs, New Jersey.",
                " In 2011 the circulation of the magazine was 1,310,696 copies.",
            ],
            [
                "The Freeway Complex Fire was a 2008 wildfire in the Santa Ana Canyon area of Orange County, California.",
                " The fire started as two separate fires on November 15, 2008.",
                ' The "Freeway Fire" started first shortly after 9am with the "Landfill Fire" igniting approximately 2 hours later.',
                " These two separate fires merged a day later and ultimately destroyed 314 residences in Anaheim Hills and Yorba Linda.",
            ],
            [
                "William Rast is an American clothing line founded by Justin Timberlake and Trace Ayala.",
                " It is most known for their premium jeans.",
                " On October 17, 2006, Justin Timberlake and Trace Ayala put on their first fashion show to launch their new William Rast clothing line.",
                " The label also produces other clothing items such as jackets and tops.",
                " The company started first as a denim line, later evolving into a men’s and women’s clothing line.",
            ],
        ],
    },
}

row2 = {
    "id": "5a879ab05542996e4f30887e",
    "question": "The Oberoi family is part of a hotel company that has a head office in what city?",
    "answer": "Delhi",
    "type": "bridge",
    "level": "medium",
    "supporting_facts": {
        "title": ["Oberoi family", "The Oberoi Group"],
        "sent_id": [0, 0],
    },
    "context": {
        "title": [
            "Ritz-Carlton Jakarta",
            "Oberoi family",
            "Ishqbaaaz",
            "Hotel Tallcorn",
            "Mohan Singh Oberoi",
            "Hotel Bond",
            "The Oberoi Group",
            "Future Fibre Technologies",
            "289th Military Police Company",
            "Glennwanis Hotel",
        ],
        "sentences": [
            [
                "The Ritz-Carlton Jakarta is a hotel and skyscraper in Jakarta, Indonesia and 14th Tallest building in Jakarta.",
                " It is located in city center of Jakarta, near Mega Kuningan, adjacent to the sister JW Marriott Hotel.",
                " It is operated by The Ritz-Carlton Hotel Company.",
                " The complex has two towers that comprises a hotel and the Airlangga Apartment respectively.",
                " The hotel was opened in 2005.",
            ],
            [
                "The Oberoi family is an Indian family that is famous for its involvement in hotels, namely through The Oberoi Group."
            ],
            [
                'Ishqbaaaz (English: "Lovers") is an Indian drama television series which is broadcast on Star Plus.',
                " It premiered on 27 June 2016 and airs Mon-Fri 10-11pm IST.",
                "Nakuul Mehta, Kunal Jaisingh and Leenesh Mattoo respectively portray Shivaay, Omkara and Rudra, the three heirs of the Oberoi family.",
                ' The show initially focused on the tale of three brothers, later become centered on the love story of Shivaay and Annika (Surbhi Chandna); with the story of Omkara and Rudra being shifted to the spinoff series "Dil Boley Oberoi".',
                ' In July 2017 "Dil Boley Oberoi" ended and the storylines were merged back into "Ishqbaaaz" which doubled its runtime.',
            ],
            [
                "The Hotel Tallcorn is located in Marshalltown, Iowa.",
                " Today it is called the Tallcorn Towers Apartments.",
                " Built in 1928 by the Eppley Hotel Company, local citizens contributed $120,000 to ensure the successful completion of this seven-story hotel.",
                " It was completed in connection to the seventy-fifth anniversary of Marshalltown.",
                " The hotel's sale in 1956 from the Eppley chain to the Sheraton Corporation was part of the second largest hotel sale in United States history.",
                " The Tallcorn was listed as a contributing property in the Marshalltown Downtown Historic District on the National Register of Historic Places in 2002.",
            ],
            [
                "Rai Bahadur Mohan Singh Oberoi (15 August 1898\xa0– 3 May 2002) was an Indian hotelier, the founder and chairman of Oberoi Hotels & Resorts, India's second-largest hotel company, with 35 hotels in India, Sri Lanka, Nepal, Egypt, Australia and Hungary."
            ],
            [
                "Hotel Bond is a historic hotel, built in two stages in 1913 and 1921, in downtown Hartford, Connecticut by hotelier Harry S. Bond.",
                " It is located near Bushnell Park, and was considered the grandest hotel in Hartford during its heyday.",
                " The second section is a 12 story building attached to the 6 story first section.",
                " A Statler Hotel opened in the area in 1954, creating competition, and the Bond Hotel company declared bankruptcy shortly after that.",
                " It was bought by the California-based Masaglia Hotel chain, which began an incremental renovation program.",
                " In 1964 it was sold to a Cincinnati, Ohio investment group which announced extensive renovation plans.",
                " However, the financing plans fell through and the hotel was again in bankruptcy.",
                " The building was sold at auction to the Roman Catholic Archdiocese of Hartford in 1965, and it became the home of the Saint Francis Hospital School of Nursing.",
                " The Bond Ballroom reopened in 2001, with the rest of the building becoming a Homewood Suites by Hilton in 2006.",
            ],
            [
                "The Oberoi Group is a hotel company with its head office in Delhi.",
                " Founded in 1934, the company owns and/or operates 30+ luxury hotels and two river cruise ships in six countries, primarily under its Oberoi Hotels & Resorts and Trident Hotels brands.",
            ],
            [
                "Future Fibre Technologies (FFT) is a fiber optic sensing technologies company based in Melbourne, Australia, with its US head office in Mountain View, California, Middle East head office in Dubai, Indian head office in New Delhi and European head office in London.",
                " Founded in 1994, Future Fibre Technologies product line provides optical fiber intrusion detection systems for perimeters, buried oil and gas pipelines and data communication networks.",
            ],
            [
                "The 289th Military Police Company was activated on 1 November 1994 and attached to Hotel Company, 3rd Infantry (The Old Guard), Fort Myer, Virginia.",
                " Hotel Company is the regiment's specialty company.",
            ],
            [
                "The Glennwanis Hotel is a historic hotel in Glennville, Georgia, Tattnall County, Georgia, built on the site of the Hughes Hotel.",
                " The hotel is located at 209-215 East Barnard Street.",
                " The old Hughes Hotel was built out of Georgia pine circa 1905 and burned in 1920.",
                " The Glennwanis was built in brick in 1926.",
                " The local Kiwanis club led the effort to get the replacement hotel built, and organized a Glennville Hotel Company with directors being local business leaders.",
                ' The wife of a local doctor won a naming contest with the name "Glennwanis Hotel", a suggestion combining "Glennville" and "Kiwanis".',
            ],
        ],
    },
}


row3 = {
    "id": "5abd90545542996e802b47d7",
    "question": "Fast Cars, Danger, Fire and Knives includes guest appearances from which hip hop record executive?",
    "answer": "Jaime Meline",
    "type": "bridge",
    "level": "medium",
    "supporting_facts": {
        "title": ["Fast Cars, Danger, Fire and Knives", "El-P"],
        "sent_id": [2, 0],
    },
    "context": {
        "title": [
            "Lights Out Paris",
            "El-P",
            "Born and Raised (EP)",
            "Lord Steppington",
            "Control Freek",
            "Hip hop",
            "Fast Cars, Danger, Fire and Knives",
            "Longterm Mentality",
            "Experimental hip hop",
            "Changes (Alyson Avenue album)",
        ],
        "sentences": [
            [
                "Lights Out Paris is the first studio album by American hip hop artist Sims, a member of Minneapolis indie hip hop collective Doomtree.",
                " It was released July 28, 2005 on Doomtree Records and includes guest appearances from P.O.S, Crescent Moon, and Toki Wright, among others.",
                ' The album was re-released with four remixes and five songs from Sims\' "False Hopes Four" on vinyl in June 2015.',
            ],
            [
                "Jaime Meline (born March 2, 1975), better known by his stage name El-P (shortened from El Producto), is an American hip hop recording artist, record producer, and record executive.",
                " Originally a member of Company Flow, El-P has been a major driving force in alternative hip hop for more than two decades, producing for several notable rappers such as Aesop Rock, Mr. Lif, and Cage, among others.",
            ],
            [
                "Born and Raised is the debut EP by American hip hop duo Smif-N-Wessun, released on December 3, 2013, under Duck Down Music Inc..",
                " Entirely produced by Beatnick & K-Salaam, the 6-song EP is a blend between reggae and hip hop, and includes guest appearances from Junior Reid, Jr.",
                " Kelly, Jahdan Blakkamoore, and DJ Full Factor.",
                ' The EP was preceded by one single — "Solid Ground" featuring dancehall icon Junior Reid.',
            ],
            [
                "Lord Steppington is the debut studio album by California-based hip hop duo Step Brothers (rapper/producers The Alchemist and Evidence).",
                " The album was released on January 21, 2014 by Rhymesayers Entertainment.",
                " The record was produced entirely by Alchemist and Evidence, and includes guest appearances from Action Bronson, Roc Marciano, Blu, Fashawn, Rakaa, Oh No, Styles P, Domo Genesis and The Whooliganz – Alchemist's old group which included actor Scott Caan.",
            ],
            [
                "Control Freek is the second solo effort by rapper Tash of the West Coast hip hop crew Tha Alkaholiks.",
                " This albums comes ten years after Tash's first well-received solo album Rap Life.",
                " It was released in 2009 on Amalgam Digital.",
                " It includes guest appearances from Tash's group Tha Alkaholiks in addition to guest spots from Del the Funky Homosapien, King T, B-Real from Cypress Hill, Knoc-turn'al, Khujo from Goodie Mob, among others.",
            ],
            [
                "Hip hop or hip-hop is a subculture and art movement developed in South Bronx in New York City during the late 1970s.",
                ' While people unfamiliar with hip hop culture often use the expression "hip hop" to refer exclusively to hip hop music (also called "rap"), Hip hop is characterized by nine distinct elements or expressive realms, of which hip hop music is only four elements (rapping, djaying, beatboxing and breaking).',
                ' Afrika Bambaataa of the hip hop collective Zulu Nation outlined the pillars of hip hop culture, coining the terms: "rapping" (also called MCing or emceeing), a rhythmic vocal rhyming style (orality); DJing (and turntablism), which is making music with record players and DJ mixers (aural/sound and music creation); b-boying/b-girling/breakdancing (movement/dance); and graffiti art, which he called "aerosol writin\'", although many say that the graffiti that hip hop adopted had been around years earlier, and had nothing to do with hip hop culture.',
                " (visual art).",
                " Other elements of hip hop subculture and arts movements beyond the main four are: hip hop culture and historical knowledge of the movement (intellectual/philosophical); beatboxing, a percussive vocal style; street entrepreneurship; hip hop language; and hip hop fashion and style, among others.",
            ],
            [
                "Fast Cars, Danger, Fire and Knives is an EP by American hip hop artist Aesop Rock.",
                " Released via the Definitive Jux label on February 22, 2005, the record is produced by Blockhead and Aesop Rock himself, with the former producing three tracks and the latter producing four, with one track produced by Rob Sonic.",
                " Vocals are handled by Aesop Rock, with guest appearances from Camu Tao and Metro of S.A. Smash and Definitive Jux label head El-P.",
                " All scratches are performed by DJ Big Wiz.",
            ],
            [
                "Longterm Mentality is the debut studio album by American hip hop recording artist Ab-Soul.",
                " It was released on April 5, 2011, by Top Dawg Entertainment (TDE), exclusively to digital retailers, serving as Ab-Soul's debut retail release.",
                " The album features guest appearances from Jhené Aiko, Schoolboy Q, Kendrick Lamar, Punch, Alori Joh, JaVonté, MURS, BJ the Chicago Kid and Pat Brown, with the production from American hip hop record producers such as Tae Beast, Ayiro, Sounwave, AAyhasis, Context, Alexis Carrington and Tommy Black.",
                " Upon its release, the album received a highly acclaimed by music critics.",
            ],
            [
                "Experimental hip hop, also known as abstract hip hop, is a genre of hip hop that employs structural elements typically considered unconventional in traditional hip hop music.",
                " Some notable experimental hip hop record labels include Definitive Jux, Anticon, Big Dada and Ninja Tune.",
                " While most experimental hip hop incorporates turntablism and is produced electronically, some artists have introduced acoustic elements to the music to facilitate it being performed live.",
            ],
            [
                "Changes is the third album by Swedish AOR/rock band Alyson Avenue with new vocalist Arabella Vitanc.",
                ' Alyson Avenue released their third album "Changes" through Avenue of Allies.',
                " The record was co-produced by band members and Chris Laney (Crazy Lixx, H.E.A.T., Brian Robertson) and includes guest appearances by Anette Olzon (Ex-Nightwish, Ex-Alyson Avenue), Michael Bormann (Ex-Jaded Heart, Charade, BISS), Rob Marcello (Danger Danger, Marcello - Vestry), Fredrik Bergh, (Street Talk, Bloodbound), Tommy Stråhle and Mike Andersson (Cloudscape, Planet Alliance).",
            ],
        ],
    },
}
