"""Main App."""
import time
import flet as ft
import folium
import webbrowser
import numpy as np

from typing import Generator
from pathlib import Path

from wildfire_detection.models.models_utils import (
    evaluate_model,
    evaluate_model_video,
    open_web_camera_with_model,
)
from wildfire_detection.data.data_utils import (
    get_coords_location,
)

PROJECT_ROOT = Path().resolve().parents[0]
MAP_PARAM = 0 # 0 is google; 1 is folium


def main(page: ft.Page) -> None:
    """Main function."""
    # ----- Main Settings ----- #
    page.route = "/"
    page.title = "Forest Fire Detection"
    page.scroll = "auto"
    page.theme_mode = ft.ThemeMode.DARK
    page.window_always_on_top = False
    page.window_center()

    # ----- Real Work Emulation ----- #
    def loader_batch_images(
        stream_data: list[Path], batch_size: int = 4, delay_time: float = 0.05,
    ) -> Generator[list[Path], None, None] | False:
        """Return batch images."""
        while True:
            data_available = len(stream_data) > 0
            if not data_available:
                return False

            batch_img = stream_data[:batch_size]
            stream_data = stream_data[batch_size:]

            yield batch_img
            time.sleep(delay_time)


    def adding_data_to_table(row_data: list) -> None:
        """Add data to table."""
        name_img = row_data[0]
        img_probs = row_data[1]
        decimal_latitude = round(row_data[2], 6)
        decimal_longitude = round(row_data[3], 6)

        has_fire = "No Fire"
        total_prob = 0.0

        clr_txt = ""
        italic_font = False
        if len(img_probs):
            has_fire = "Yes Fire"
            total_prob = 1 - np.prod([1 - prob for prob in img_probs])
            total_prob = round(total_prob, 2)

            italic_font = True
            clr_txt = "red400"


        stats_table.rows.append(
            ft.DataRow(
                cells=[
                    ft.DataCell(ft.Text(
                        name_img, color=clr_txt, size=16, italic=italic_font,
                    )),
                    ft.DataCell(ft.Text(
                        has_fire, color=clr_txt, size=16, italic=italic_font,
                    )),
                    ft.DataCell(ft.Text(
                        total_prob, color=clr_txt, size=16, italic=italic_font,
                    )),
                    ft.DataCell(ft.Text(
                        decimal_latitude, color=clr_txt, size=16, italic=italic_font,
                    )),
                    ft.DataCell(ft.Text(
                        decimal_longitude, color=clr_txt, size=16, italic=italic_font,
                    )),
                ],
            ),
        )


    def run_real_work(e: ft.ControlEvent) -> None:
        """Run real work."""
        data_dpath = PROJECT_ROOT / "data" / "test"
        stream = list(data_dpath.glob("*.jpg"))

        batch_imgs = loader_batch_images(stream)
        row_container.content.controls[1] = row_image_holder
        page.update()

        map_row.visible = True
        fol_map = folium.Map()
        map_save_path = PROJECT_ROOT / "data" / "map_data"

        global MAP_PARAM
        MAP_PARAM = 1

        total_num_data = 0

        while True:
            imgs_lst = next(batch_imgs, False)
            if imgs_lst is False:
                break

            batch_probs = evaluate_model(imgs_lst)
            total_num_data += len(imgs_lst)

            for img_fpath, img_probs in zip(imgs_lst, batch_probs):
                # Predict with model
                path_img = PROJECT_ROOT / "data" / "predicted" / img_fpath.name
                row_image_holder.content.controls.append(
                    ft.Image(
                        src=path_img,
                        visible=True,
                        fit=ft.ImageFit.CONTAIN,
                        repeat=ft.ImageRepeat.NO_REPEAT,
                        border_radius=ft.border_radius.all(10),
                    )
                )

                # Add points on map
                decimal_latitude, decimal_longitude = get_coords_location(img_fpath)
                folium.Marker(
                    location=[decimal_latitude, decimal_longitude],
                    tooltip="Click for more information",
                ).add_to(fol_map)

                fol_map.fit_bounds(fol_map.get_bounds())
                fol_map.save(map_save_path / "map.html")

                # Add data to table
                adding_data_to_table([
                    path_img.name,
                    img_probs,
                    decimal_latitude,
                    decimal_longitude,
                ])

                page.update()

    # ----- Functions ----- #
    def web_camera_clicked(e: ft.ControlEvent) -> None:
        """Open Web Camera with model evaluating."""
        open_web_camera_with_model()


    def map_btn_clicked(e: ft.ControlEvent) -> None:
        """Open map in browser."""
        if MAP_PARAM:
            map_path = PROJECT_ROOT / "data" / "map_data" / "map.html"
            webbrowser.open("file://" + str(map_path), new=2)
            return

        current_img_fpath = image_holder.content.src
        img_fpath = current_img_fpath.resolve().parents[1] / "test" / current_img_fpath.name

        decimal_latitude, decimal_longitude = get_coords_location(img_fpath)
        url = f"https://www.google.com/maps?q={decimal_latitude},{decimal_longitude}"
        webbrowser.open_new_tab(url)


    def run_predict(e: ft.ControlEvent) -> None:
        """Evaluate model and show image."""
        path_file = Path(image_holder.content.src)
        if "mp4" in path_file.name:
            evaluate_model_video(path_file)
            return

        image_holder.content = ft.Container(
            ft.Column([
                ft.Text("Evaluating model..."),
                ft.ProgressBar(width=400, color="amber", bgcolor="#eeeeee"),
            ]),
            alignment=ft.alignment.bottom_center,
            padding=ft.padding.only(top=page.window_height / 2.5),
        )
        page.update()

        probs = evaluate_model(path_file)
        predicted_data = PROJECT_ROOT / "data" / "predicted"
        predicted_data /= path_file.name

        image_holder.content = ft.Image(
            src=predicted_data,
            visible=True,
            fit=ft.ImageFit.CONTAIN,
        )
        map_row.visible = True

        page.update()


    def img_btn_clicked(e: ft.ControlEvent) -> None:
        """Show image."""
        image_holder.content.src = e.control.data
        image_holder.content.visible = True

        run_predictor_btn.color = "#FF6E1C"
        run_predictor_btn.bgcolor = "#8E0A3D"
        run_predictor_btn.disabled = False

        for item in column_data.content.controls:
            item.content.icon_color = "green600"
            if item.content.text == e.control.data.name:
                item.content.icon_color = "red500"

        page.update()


    def container_item(lst: list[Path]) -> list[ft.Container]:
        """Add list of items to container."""
        names_lst = [path_file.name for path_file in lst]
        items = []

        for item, path in zip(names_lst, lst):
            items.append(
                ft.Container(
                    content=ft.TextButton(
                        text=str(item),
                        on_click=img_btn_clicked,
                        data=path,
                        icon="park_rounded",
                        icon_color="green600",
                    ),
                    alignment=ft.alignment.center_left,
                    bgcolor="#FFFFF",
                    border_radius=ft.border_radius.all(5),
                    padding=ft.padding.all(2),
                )
            )

        return items


    def handle_loaded_file(e: ft.FilePickerResultEvent) -> None:
        """File Picker."""
        files_dir = Path(e.files[0].path).resolve().parents[0]
        files_path = []

        # Get list files
        for ext in allowed_ext_files:
            files_path.extend(files_dir.glob(f"*.{ext}"))

        column_data.clean()
        column_data.content.controls = container_item(files_path)
        column_data.visible = True

        page.update()


    # ----- Top Row Of Buttons ----- #
    allowed_ext_files = ["jpg", "png", "jpeg", "mp4"]
    select_image_btn = ft.ElevatedButton(
        text="Open Gallery",
        bgcolor="#019267",
        on_click=lambda _: file_picker.pick_files(
            allow_multiple=False,
            allowed_extensions=allowed_ext_files,
        ),
    )

    run_predictor_btn = ft.ElevatedButton(
        text="Run Prediction",
        on_click=run_predict,
        disabled=True,
    )

    main_buttons = ft.Container(
        ft.Row(
            [select_image_btn, run_predictor_btn],
            spacing=20,
        ),
    )

    text_map = ft.Text("Open Map Location:")
    map_button = ft.IconButton(
        icon=ft.icons.MAP_SHARP,
        on_click=map_btn_clicked,
        icon_color="red600",
        icon_size=30,
        tooltip="Open Map",
    )

    map_row = ft.Container(
        ft.Row(
            [text_map, map_button],
            spacing=5,
        ),
        visible=False,
    )

    popup_menu_btn = ft.Container(
        ft.PopupMenuButton(
            items=[
                ft.PopupMenuItem(
                    icon=ft.icons.ARCHITECTURE,
                    text="Open Demonstrate Page",
                    on_click=None,
                ),
                ft.PopupMenuItem(),
                ft.PopupMenuItem(
                    icon=ft.icons.CAMERA,
                    text="Open Web Camera",
                    on_click=web_camera_clicked,
                ),
                ft.PopupMenuItem(),
                ft.PopupMenuItem(
                    icon=ft.icons.FOREST_SHARP,
                    text="Run Real Work",
                    on_click=run_real_work,
                ),
            ],
        ),
    )

    buttons_row = ft.Row(
        controls=[
            main_buttons,
            map_row,
            popup_menu_btn,
        ],
        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
    )


    # ----- List And Image ----- #
    file_picker = ft.FilePicker(on_result=handle_loaded_file)
    page.overlay.append(file_picker)

    column_data = ft.Container(
        content=ft.Column(
            spacing=10,
            height=100,
            width=100,
            scroll=ft.ScrollMode.ALWAYS,
        ),
        gradient=ft.LinearGradient(
            begin=ft.alignment.top_center,
            end=ft.alignment.bottom_center,
            colors=["#019267", "#4F4E4E"],
            rotation=45,
        ),
        width=300,
        height=page.window_height - 50,
        border_radius=ft.border_radius.all(5),
        padding=ft.padding.all(10),
        visible=True,
    )

    txt_for_column_data = ft.Text(
        "Click on 'Open Gallery'",
        visible=True,
        size=16,
        color="purple100",
    )
    column_data.content.controls.append(txt_for_column_data)

    image_holder = ft.Container(
        content=ft.Image(
            visible=False,
            fit=ft.ImageFit.CONTAIN,
        ),
        width=page.window_width - column_data.padding.left - 90,
        height=page.window_height - 15,
        alignment=ft.alignment.center_left,
        border_radius=ft.border_radius.all(5),
    )

    row_image_holder = ft.Container(
        content=ft.Row(
            expand=1,
            wrap=False,
            scroll=ft.ScrollMode.ALWAYS,
        ),
        width=page.window_width - column_data.padding.left - 90,
        height=page.window_height - 15,
        alignment=ft.alignment.center_left,
        border_radius=ft.border_radius.all(5),
    )

    row_container = ft.Container(
        content=ft.Row(
            controls=[column_data, image_holder],
            spacing=20,
        ),
    )

    # ----- Statistics ----- #
    stats_text = ft.Text(
        "Results Based On Data",
        size=18,
        color="deeppurple100",
    )

    stats_table = ft.DataTable(
        columns=[
            ft.DataColumn(
                ft.Text("Image Name", size=18),
            ),
            ft.DataColumn(
                ft.Text("Fire", size=18),
            ),
            ft.DataColumn(
                ft.Text("Total Fire Probability", size=18),
                numeric=True,
            ),
            ft.DataColumn(
                ft.Text("Latitude", size=18),
                numeric=True
            ),
            ft.DataColumn(
                ft.Text("Longitude", size=18),
                numeric=True,
            ),
        ],
        rows=[],
        border_radius=ft.border_radius.all(5),
        heading_row_color="indigo400",
    )

    col_stats = ft.Container(
        content=ft.Column(
            controls=[stats_table],
            scroll=ft.ScrollMode.ALWAYS,
        ),
        height=500,
        margin=ft.margin.only(bottom=50),
    )


    # ----- Adding Elements ----- #
    page.add(buttons_row)
    page.add(row_container)
    page.add(stats_text)
    page.add(col_stats)

    page.update()


ft.app(target=main)
