"""Petit corpus d'évaluation (français) avec requêtes et document attendu."""

DOCS = {
    "ia.txt": "L'intelligence artificielle regroupe des techniques permettant à des machines de simuler des capacités humaines. "
              "L'apprentissage automatique est une branche de l'intelligence artificielle fondée sur des données. "
              "Les réseaux de neurones profonds ont permis des avancées majeures en vision par ordinateur et en traitement du langage.",
    "climat.txt": "Le réchauffement climatique est causé par l'augmentation des gaz à effet de serre, notamment le dioxyde de carbone. "
                  "Les océans absorbent la chaleur et le niveau de la mer monte. Les accords de Paris visent à limiter la hausse des températures.",
    "cuisine.txt": "Pour préparer une ratatouille, coupez les aubergines, les courgettes et les poivrons. Faites revenir les légumes dans l'huile d'olive "
                   "avec des oignons et de l'ail, puis laissez mijoter une heure. La cuisson lente développe les saveurs.",
    "sante.txt": "Une alimentation équilibrée et une activité physique régulière réduisent le risque de maladies cardiovasculaires. "
                 "Le sommeil et la gestion du stress jouent aussi un rôle important pour la santé.",
    "football.txt": "Le football se joue entre deux équipes de onze joueurs. Le gardien de but est le seul à pouvoir toucher le ballon avec les mains. "
                    "La Coupe du monde réunit les meilleures sélections nationales tous les quatre ans.",
    "histoire.txt": "La Révolution française débute en 1789 avec la prise de la Bastille. La Déclaration des droits de l'homme et du citoyen "
                    "proclame l'égalité des citoyens. La monarchie est abolie en 1792.",
    "energie.txt": "Les énergies renouvelables comme le solaire et l'éolien remplacent progressivement les combustibles fossiles. "
                   "Les panneaux photovoltaïques transforment la lumière du soleil en électricité.",
    "bd.txt": "Les bases de données relationnelles organisent les informations en tables liées par des clés. Le langage SQL permet d'interroger "
              "et de modifier ces données. L'indexation accélère les requêtes.",
    "voyage.txt": "Pour voyager à l'étranger, vérifiez la validité de votre passeport et réservez vos billets d'avion à l'avance. "
                  "Pensez à souscrire une assurance voyage et à changer des devises.",
    "finance.txt": "Les marchés financiers permettent d'échanger des actions et des obligations. L'inflation réduit le pouvoir d'achat, "
                   "et les banques centrales ajustent les taux d'intérêt pour la contrôler.",
}

# (requête, document attendu)
QUERIES = [
    ("apprentissage machine", "ia.txt"),
    ("machines qui apprennent", "ia.txt"),
    ("intelligence artificielle", "ia.txt"),
    ("rechauffement climatique", "climat.txt"),          # sans accent
    ("temperature de la planète", "climat.txt"),
    ("gaz carbonique", "climat.txt"),
    ("recette légumes mijotés", "cuisine.txt"),
    ("cuisiner des courgettes", "cuisine.txt"),
    ("risque cardiaque", "sante.txt"),
    ("manger équilibré", "sante.txt"),
    ("gardien de but", "football.txt"),
    ("joueurs de foot", "football.txt"),
    ("revolution francaise", "histoire.txt"),
    ("prise de la bastille", "histoire.txt"),
    ("energie solaire", "energie.txt"),
    ("panneau photovoltaique", "energie.txt"),
    ("requete SQL", "bd.txt"),
    ("base de donnée", "bd.txt"),
    ("intelligense artificielle", "ia.txt"),             # faute de frappe
    ("climatique rechauffment", "climat.txt"),           # faute de frappe
    ("passeport avion", "voyage.txt"),
    ("taux d'intérêt banque", "finance.txt"),
    ("inflation", "finance.txt"),
    ("les", None),                                       # que des mots vides -> aucun résultat
]
