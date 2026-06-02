    label = os.path.splitext(os.path.basename(csv_path))[0]
        x_values, y_values = load_csv(csv_path, temp_offset=10)
        plt.plot(x_values, y_values, label=label)