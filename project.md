polymer_generator/
│
├── polymer_lib/                    # Основная библиотека
│   ├── __init__.py                 # Экспорт Monomer и Polymerizer
│   │
│   ├── calc.py                     # Математические функции (geometry)
│   │   ├── angle_vec()             # Угол между векторами
│   │   ├── rotate_rod()            # Поворот координат вокруг оси
│   │   ├── dihedral_coord()        # Вычисление диэдрального угла
│   │   └── distance_matrix()       # Матрица расстояний между атомами
│   │
│   ├── utils.py                    # Утилиты работы с молекулами
│   │   ├── radon_print()           # Логгер с уровнями
│   │   ├── deepcopy_mol()          # Глубокое копирование Mol
│   │   ├── remove_atom()           # Удаление атома из Mol
│   │   ├── add_bond()              # Добавление связи в Mol
│   │   ├── star2h()                # Конвертация '*' -> '[3H]'
│   │   ├── h2star()                # Конвертация '[3H]' -> '*'
│   │   └── mol_from_smiles()       # SMILES -> Mol с 3D координатами
│   │
│   ├── poly.py                     # Функции полимеризации (ядро)
│   │   ├── set_linker_flag()       # Поиск head/tail линкеров
│   │   ├── combine_mols()          # Объединение двух молекул
│   │   ├── connect_mols()          # Соединение mol1.tail -> mol2.head
│   │   ├── check_3d_proximity()    # Проверка стерических конфликтов
│   │   ├── check_3d_bond_length()  # Проверка длин связей
│   │   └── check_3d_structure_poly() # Комплексная проверка структуры
│   │
│   ├── monomer.py                  # Класс мономера
│   │   └── Monomer
│   │       ├── __init__(smiles)    # Парсинг SMILES + 3D генерация
│   │       └── copy()              # Глубокое копирование мономера
│   │
│   └── polymerizer.py              # Класс контроллера полимеризации
│       └── Polymerizer
│           ├── __init__(monomers, ...)  # Настройка параметров
│           └── build()                  # Запуск процесса с retry/rollback
│
└── example/                        # Примеры использования
    └── new_api.py                      # Тестовый скрипт (полистирол, n=50)

new_api.py
  └── Polymerizer (polymerizer.py)
        ├── Monomer (monomer.py)
        │     └── utils.mol_from_smiles()
        │
        ├── poly.connect_mols()
        │     ├── poly.set_linker_flag()
        │     ├── poly.combine_mols()
        │     ├── utils.remove_atom()
        │     ├── utils.add_bond()
        │     └── calc.* (angle_vec, rotate_rod, dihedral_coord)
        │
        └── poly.check_3d_structure_poly()
              ├── poly.check_3d_proximity()
              └── poly.check_3d_bond_length()