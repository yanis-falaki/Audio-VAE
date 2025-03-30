import torch
import matplotlib.pyplot as plt
import numpy as np


def display_examples(model, dataset, device, num_examples=30, pairs_per_row=2):
    rows = (num_examples + pairs_per_row - 1) // pairs_per_row  # Ceiling division: 2 rows for 5 examples
    
    model.eval()

    with torch.no_grad():
        # Create a figure with rows and 4 columns
        fig, axes = plt.subplots(rows, 4, figsize=(12, 3 * rows))  # 4 columns for 2 pairs
        
        for i in range(num_examples):
            # Calculate row and column indices
            row_idx = i // pairs_per_row
            pair_idx = i % pairs_per_row
            col_idx = pair_idx * 2  # 0 or 2 for each pair’s original

            # Get the original image and label
            image, label = dataset[i]
            image = image.to(device).unsqueeze(dim=0)

            # Get the reconstructed image
            reconstructed_image = model(image)

            # Move to CPU and squeeze
            image = image.squeeze().to('cpu')
            reconstructed_image = reconstructed_image.squeeze().to('cpu')

            # Plot original image
            axes[row_idx, col_idx].imshow(image, cmap='gray')
            axes[row_idx, col_idx].set_title(f"Orig (Label: {label})")
            axes[row_idx, col_idx].axis('off')

            # Plot reconstructed image
            axes[row_idx, col_idx + 1].imshow(reconstructed_image, cmap='gray')
            axes[row_idx, col_idx + 1].set_title("Recon")
            axes[row_idx, col_idx + 1].axis('off')

        # Hide empty subplots (if num_examples doesn’t fill all columns)
        for i in range(num_examples, rows * pairs_per_row):
            row_idx = i // pairs_per_row
            col_idx = (i % pairs_per_row) * 2
            axes[row_idx, col_idx].axis('off')      # Hide original slot
            axes[row_idx, col_idx + 1].axis('off')  # Hide reconstructed slot

        # Adjust layout and display
        plt.tight_layout()
        plt.show()


# Function to collect latent vectors and labels from multiple batches
def collect_latents(model, data_loader, device, num_samples=3000):
    all_latents = []
    all_labels = []
    samples_collected = 0
    
    # Set model to evaluation mode
    model.eval()
    
    with torch.no_grad():
        for images, labels in data_loader:
            if samples_collected >= num_samples:
                break
                
            # Move batch to device
            images = images.to(device)
            
            # Get encodings
            _ = model(images)
            batch_latents = model.latent
            
            # Move to CPU and convert to numpy
            batch_latents_np = batch_latents.cpu().numpy()
            batch_labels_np = labels.cpu().numpy()
            
            # Collect only what we need to reach num_samples
            remaining = num_samples - samples_collected
            if len(batch_labels_np) > remaining:
                batch_latents_np = batch_latents_np[:remaining]
                batch_labels_np = batch_labels_np[:remaining]
            
            all_latents.append(batch_latents_np)
            all_labels.append(batch_labels_np)
            samples_collected += len(batch_labels_np)
            
    # Combine all batches
    latents_np = np.vstack(all_latents)
    labels_np = np.concatenate(all_labels)
    
    return latents_np, labels_np


def display_latents(model, dataset, device, num_samples=3000):
    # Collect ~3000 points
    latents_np, labels_np = collect_latents(model, dataset, device, num_samples)

    # Set up the plot
    plt.figure(figsize=(12, 10))

    # Create a scatter plot with points colored by digit label
    scatter = plt.scatter(latents_np[:, 0], latents_np[:, 1], 
                        c=labels_np, alpha=0.7, s=20, 
                        cmap='tab10', edgecolors='none')

    # Add color bar to show the mapping of colors to digit labels
    cbar = plt.colorbar(scatter, ticks=np.arange(10))
    cbar.set_label('Digit Labels')

    # Add labels and title
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title('2D Latent Space Visualization (3000 points) Colored by Digit Class (0-9)')

    # Add a legend with digit classes
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                markerfacecolor=scatter.cmap(scatter.norm(i)), 
                                markersize=10, label=f'Digit {i}') 
                    for i in range(10)]
    plt.legend(handles=legend_elements, loc='best')

    # Display the plot
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()