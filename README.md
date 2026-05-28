# MOI - LR3 Path Tracing

Репозиторий содержит минимальный физически корректный рендерер глобального освещения методом трассировки путей для треугольных сеток.

## Что реализовано

- Геометрия сцены на треугольниках.
- Ручная сцена (аналог Cornell box) и загрузка треугольной модели из OBJ.
- RGB-цветовая модель.
- Материалы: диффузный Ламберт + зеркальный компонент (чистое зеркало), одновременно на одной поверхности.
- Физическое ограничение материалов: `kd + ks <= 1` по каждой компоненте.
- Выбор события рассеяния методами русской рулетки и importance sampling.
- Трассировка путей с глобальным освещением и next event estimation:
	- один случайный луч на прямой свет,
	- случайный выбор источника по мощности.
- Камера-точка, случайный джиттер внутри пикселя (anti-aliasing).
- Разрешение задается параметрами, по умолчанию `600x600`.
- Нормировка яркостей (`max`, `mean`, `fixed`), клиппинг, гамма-коррекция и запись в PPM.
- Дополнительно: запись линейного HDR-подобного буфера в PFM.

## Файл реализации

- `path_tracer.py` - основной исполняемый файл.
- `lr4_path_tracer (2).py` - GUI-версия ЛР4 с BRDF `lambert/cook_torrance`, AOV-буферами и сохранением `.npz`.
- `lr5_filters.py` - отдельный скрипт ЛР5: `gaussian`, `bilateral`, `multilateral`, `median` фильтрация по AOV.

## Быстрый запуск

```bash
/Users/kdo/Desktop/MOI/.venv/bin/python path_tracer.py \
	--scene manual \
	--width 600 --height 600 \
	--spp 64 --max-depth 8 \
	--output renders/manual.ppm \
	--pfm-output renders/manual.pfm
```

## Запуск со сценой из OBJ

```bash
/Users/kdo/Desktop/MOI/.venv/bin/python path_tracer.py \
	--scene obj \
	--obj-path /absolute/path/to/model.obj \
	--width 600 --height 600 \
	--spp 64 --max-depth 8 \
	--output renders/obj.ppm
```

## Основные параметры CLI

- `--scene {manual,obj}`: тип сцены.
- `--obj-path`: путь к OBJ (обязательно для `--scene obj`).
- `--width`, `--height`: разрешение.
- `--spp`: число лучей на пиксель.
- `--max-depth`: максимальная глубина пути.
- `--max-seconds`: ограничение по времени рендера (секунды).
- `--seed`: зерно ГПСЧ.
- `--normalization {max,mean,fixed}`: способ нормировки.
- `--fixed-scale`: масштаб для режима `fixed`.
- `--gamma`: коэффициент гамма-коррекции.
- `--output`: итоговый LDR-файл в PPM (`P6`).
- `--pfm-output`: необязательный линейный буфер в PFM.

## ЛР5: запуск фильтрации

1. Сначала в `lr4_path_tracer (2).py` выполните рендер и нажмите **Сохранить AOV (.npz)**.
2. Затем запустите:

```bash
python3 lr5_filters.py \
  --input-aov /absolute/path/to/render_aov.npz \
  --out-dir renders/lr5 \
  --radius 3 --sigma-spatial 2.0 --sigma-color 0.2 \
  --sigma-depth 0.2 --sigma-normal 0.2 \
  --normalize p99 --gamma 2.2
```

## Проверка требований задания

- Минимальное итоговое разрешение для отчета: не меньше `500x500`.
- Для защиты можно менять параметры сцены и рендера через CLI без изменения кода.
- Подробное описание алгоритма и корректности приведено в `REPORT.md`.
# moi-5
